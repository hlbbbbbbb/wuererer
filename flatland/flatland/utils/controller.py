
import copy
from http.client import UnimplementedFileMode
from threading import local
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator, rail_from_file
from flatland.envs.schedule_generators import complex_schedule_generator, schedule_from_file
from flatland.envs.malfunction_generators import ParamMalfunctionGen,MalfunctionParameters,malfunction_from_file

from flatland.envs.rail_env import RailEnv
from enum import IntEnum
import time, os, sys, json, argparse, glob
import numpy as np

parser = argparse.ArgumentParser(description='Args for remote evaluation')
parser.add_argument('--remote-mode', default = False, action="store_true",
                    help='If running in remote mode')
parser.add_argument('--tests', type=str, default = None,
                    help='Path for test cases')
parser.add_argument('-q', type=int, default = 1,
                    help='Question type')      
parser.add_argument('-o', type=str, default = None,
                    help='Output file')                

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

_original_stdout = sys.stdout
def mute_print():
    _original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

def unmute_print():
    sys.stdout.close()
    sys.stdout = _original_stdout

def eprint(*args, **kwargs):
    print("[ERROR] ",*args, file=sys.stderr, **kwargs)

def wprint(*args, **kwargs):
    print("[WARN] ",*args, file=sys.stderr, **kwargs)


output_template = "{0:18} | {1:12} | {2:12} | {3:12} | {4:10} | {5:12} | {6:12} | {7:12} | {8:12} | {9:12}"
csv_template = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n"

output_header = output_template.format("Test case", "Total agents","Agents done", "DDLs met","Plan Time", "SIC", "Makespan","Penalty","Final SIC","P Score")


class Train_Actions(IntEnum):
    NOTHING = 0
    LEFT = 1
    FORWARD = 2
    RIGHT = 3
    STOP = 4

class Directions(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


def path_controller(time_step,local_env: RailEnv, path_all: list, debug=False):
    action_dict = {}
    out_of_path = True
    inconsistent = False
    for agent_id in range(0, len(local_env.agents)):
        if time_step == 0:
            if len(path_all[agent_id]) >0:
                action_dict[agent_id] = Train_Actions.FORWARD
            else:
                action_dict[agent_id] = Train_Actions.NOTHING
            out_of_path = False
        elif time_step >= len(path_all[agent_id]) or local_env.agents[agent_id].status==3:
            action_dict[agent_id] = Train_Actions.NOTHING
        else:
            action_dict[agent_id] = get_action(agent_id, path_all[agent_id][time_step], local_env)
            if action_dict[agent_id] == -1:
                action_dict[agent_id] = Train_Actions.STOP
                if debug:
                    eprint("Agent {} cannot reach location {} from location {}. Path is inconsistent." \
                       .format(agent_id, path_all[agent_id][time_step], local_env.agents[agent_id].position))
                inconsistent = True

            out_of_path = False
    return action_dict,out_of_path,inconsistent

def get_action(agent_id: int, next_loc: tuple, env: RailEnv):
    current_loc = env.agents[agent_id].position
    current_direction = env.agents[agent_id].direction
    if  current_loc== next_loc:
        return Train_Actions.STOP

    move_direction = 0
    if next_loc[0] - current_loc[0] == 1:
        move_direction = Directions.SOUTH
    elif next_loc[0] - current_loc[0] == -1:
        move_direction = Directions.NORTH
    elif next_loc[1] - current_loc[1] == -1:
        move_direction = Directions.WEST
    elif next_loc[1] - current_loc[1] == 1:
        move_direction = Directions.EAST
    else:
        move_direction = -1

    if move_direction == -1:
        return -1

    if move_direction == current_direction:
        return Train_Actions.FORWARD
    elif move_direction - current_direction == 1 or move_direction - current_direction == -3 :
        return Train_Actions.RIGHT
    elif move_direction - current_direction == -1 or move_direction - current_direction == 3 :
        return Train_Actions.LEFT
    elif move_direction - current_direction == 2 or move_direction - current_direction == -2:
        return Train_Actions.FORWARD

    return -1

def check_conflict(time_step,path_all,local_env: RailEnv, debug=False):
    conflict = False
    failed_agents = []
    for agent_id in range(0, len(local_env.agents)):
        if local_env.agents[agent_id].position != None and len(path_all[agent_id]) > time_step and local_env.agents[agent_id].position != path_all[agent_id][time_step]:
            conflict_id = -1
            failed_agents.append(agent_id)
            for i in range(0, len(local_env.agents)):
                if i != agent_id and  path_all[agent_id][time_step] == local_env.agents[i].position:
                    conflict_id = i
                    

            if debug:
                if conflict_id == -1:
                    wprint("Agent {} failed to move to {} at timestep {}. Will call replan function if in question 3.".format(agent_id, path_all[agent_id][time_step], time_step))
                else:
                    wprint("Agent {} have conflict when trying to reach {} at timestep {} with Agent {}. Will call replan function if in question 3.".format(agent_id, path_all[agent_id][time_step], time_step,conflict_id))
            conflict = True
    return conflict, failed_agents

def evaluator(get_path, test_cases: list, debug: bool, visualizer: bool, question_type: int, 
              ddl: list=None, ddl_scale: int=0.2, baseline_pscore = {}, save_pscore = None,  
              penalty_scale=2, mute = False, write = None, replan = None):
    statistics = []
    runtimes = []
    pscores = {}
    if visualizer:
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
    print(output_header, flush=True)
    if write is not None:
        out = open(write, "w+",1)
    
    for i, test_case in enumerate(test_cases):
        test_name =  os.path.basename(os.path.dirname(test_case))+"/"+os.path.basename(test_case).replace(".pkl","")
        if debug:
            print("Loading evaluation: {}".format(test_case))
        local_env = RailEnv(width=1,
                            height=1,
                            rail_generator=rail_from_file(test_case),
                            schedule_generator=schedule_from_file(test_case),
                            # schedule_generator=schedule_from_file(test_case, ddl_test_case),
                            remove_agents_at_target=True,
                            malfunction_generator_and_process_data= malfunction_from_file(test_case) if question_type == 3 else None
                            # Removes agents at the end of their journey to make space for others
                            )

        local_env.reset()

        num_of_agents = local_env.get_num_agents()
        statistic_dict = {"test_case": test_name,"No. of agents":local_env.get_num_agents(), "time_step": 0, "num_done": 0, "deadlines_met": 0, "sum_of_cost": 0, "done_percentage": 0,
                          "all_done": False, "cost":[0] * num_of_agents,"penalty":[0]*num_of_agents,"sic_final":[0]*num_of_agents,"p":0,"f":0}

        # Initiate the renderer
        if visualizer:
            env_renderer = RenderTool(local_env,
                                      show_debug=True,
                                      screen_height=900,  # Adjust these parameters to fit your resolution
                                      screen_width=900)  # Adjust these parameters to fit your resolution
            env_renderer.render_env(show=True, show_observations=False, show_predictions=False)
        path_all = []
        start_t = time.time()
        if mute:
            mute_print()
        if question_type == 1:
            agent_id = 0
            agent = local_env.agents[agent_id]
            path = get_path(agent.initial_position, agent.initial_direction, agent.target, copy.deepcopy(local_env.rail), local_env._max_episode_steps)
            path_all.append(path[:])
            if debug:
                print("Agent: {}, Path: {}".format(agent_id, path))
        elif question_type == 2:
            for agent_id in range(0, len(local_env.agents)):
                agent = local_env.agents[agent_id]
                path = get_path(agent.initial_position, agent.initial_direction, agent.target, copy.deepcopy(local_env.rail), agent_id,
                                path_all[:], local_env._max_episode_steps)
                path_all.append(path[:])
                if debug:
                    print("Agent: {}, Path: {}".format(agent_id, path))
        elif question_type == 3:
            if ddl:
                deadlines = local_env.read_deadlines(ddl[i])
            else:
                expected_delay = local_env.malfunction_process_data.malfunction_rate*(local_env.width+local_env.height)*((local_env.malfunction_process_data.min_duration+local_env.malfunction_process_data.max_duration)/2)
                deadlines = local_env.generate_deadlines(ddl_scale, 
                                                         group_size= max(1,len(local_env.agents)//5), 
                                                         malfunction_scale=(1 + expected_delay/(local_env.width+local_env.height)/2 ))
                local_env.save_deadlines(test_case[:-4], deadlines)
            local_env.set_deadlines(deadlines)

            path_all = get_path(copy.deepcopy(local_env.agents), copy.deepcopy(local_env.rail), local_env._max_episode_steps)
            if debug:
                for agent_id in range(0, len(local_env.agents)):
                    print("Agent: {}, Path: {}".format(agent_id, path_all[agent_id]))

        else:
            eprint("No such question type option.")
            exit(1)
        if mute:
            unmute_print()
        runtimes.append(round(time.time()-start_t,2))

        replan_runtime = 0
        time_step = 0
        out_of_path = False
        inconsistent = False
        done = None
        while time_step < local_env._max_episode_steps:
            if out_of_path:
                if debug:
                    eprint("Reach last location in all paths. Current timestep: {}".format(time_step))
                    eprint("Can't finish test {}.".format(test_case))
                    eprint("Press Enter to move to next test:")
                    input()
                break

            if inconsistent:
                if debug:
                    wprint("Press Enter to continue:")
                    input()

            action_dict, out_of_path, inconsistent = path_controller(time_step, local_env, path_all, debug)
            statistic_dict["time_step"] = time_step


            malfunction_before = [agent.malfunction_data["malfunction"]>0 and agent.status < 2 for agent in local_env.agents]
            # execuate action
            next_obs, all_rewards, done, _ = local_env.step(action_dict)

            if visualizer:
                env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

            # Find malfunction and failed execuation agents. Then call replan function.
            if question_type == 3:
                conflict = False
                failed_agents = []
                new_malfunctions = []
                if time_step!=0:
                    conflict, failed_agents = check_conflict(time_step, path_all, local_env, debug)
                    if conflict:
                        if debug:
                            wprint("Press Enter to continue:")
                            input()
                for agent in local_env.agents:
                    if agent.status !=3 and time_step >= len(path_all[agent.handle]):
                        failed_agents.append(agent.handle)
                    if agent.malfunction_data["malfunction"]>0 and agent.status < 2 and not malfunction_before[agent.handle]:
                        new_malfunctions.append(agent.handle)
                if len(new_malfunctions) != 0 or conflict:
                    if debug:
                        print("Find new malfunctions: ", new_malfunctions, " Find failed execuation agents: ", failed_agents)
                        print("Call replan function... ...")
                    replan_start = time.time()
                    if mute:
                        mute_print()
                    new_paths = replan(copy.deepcopy(local_env.agents), copy.deepcopy(local_env.rail), time_step, copy.deepcopy(path_all), local_env._max_episode_steps, new_malfunctions, failed_agents)
                    if mute:
                        unmute_print()
                    replan_runtime += round(time.time()-replan_start,2)
                    path_all = new_paths


            num_done = 0
            new_cost = 0
            num_deadlines_met = 0
            for agent_id in range(0, len(local_env.agents)):
                if local_env.agents[agent_id].status in [2,3] :
                    num_done += 1
                    if question_type == 3 and local_env.agents[agent_id].deadline:
                        if statistic_dict["cost"][agent_id] <= local_env.agents[agent_id].deadline:
                            num_deadlines_met += 1
                    else:
                        num_deadlines_met += 1
                else:
                    if question_type == 3 and time_step > local_env.agents[agent_id].deadline:
                        statistic_dict["penalty"][agent_id]+=1
                    statistic_dict["cost"][agent_id]+= 1

            statistic_dict["num_done"] = num_done
            statistic_dict["done_percentage"] = round(num_done / len(local_env.agents), 2)
            statistic_dict["deadlines_met"] = num_deadlines_met

            if debug:
                time.sleep(0.2)


            if (done["__all__"]):
                statistic_dict["all_done"] = True
                if debug:
                    print("All agents reach destination at timestep: {}.  Move to next test in 1 seconds ...".format(time_step))
                time.sleep(1)
                break
            time_step += 1
        
        runtimes[-1] += replan_runtime
        # End of one episode. 
        for agent_id in range(0, len(local_env.agents)):
            if done[agent_id]:
                statistic_dict["sum_of_cost"] += statistic_dict["cost"][agent_id]
            else:
                statistic_dict["sum_of_cost"] += local_env._max_episode_steps
        statistic_dict["sic_final"] = statistic_dict["sum_of_cost"] + sum(statistic_dict["penalty"])
        if question_type == 1:
            statistic_dict["p"] = None
        else:
            statistic_dict["p"] = int(statistic_dict["sic_final"]/num_of_agents)
            if baseline_pscore:
                statistic_dict["f"] =min(round(baseline_pscore[test_case]/statistic_dict["p"],2),1.0)
        pscores[test_case] = statistic_dict["p"]
        print(output_template.format(test_name, str(statistic_dict["No. of agents"]), str(statistic_dict["num_done"]),
                                     str(statistic_dict["deadlines_met"]), str(runtimes[-1]),
                                     str(statistic_dict["sum_of_cost"]), str(statistic_dict["time_step"]),
                                     str(sum(statistic_dict["penalty"])),str(statistic_dict["sic_final"]),str(statistic_dict["p"])+("({})".format(statistic_dict["f"]) if baseline_pscore else "")
                                     ),flush=True)
        if write is not None:
            out.write(csv_template.format(test_name, str(statistic_dict["No. of agents"]), str(statistic_dict["num_done"]),
                                     str(statistic_dict["deadlines_met"]), str(runtimes[-1]),
                                     str(statistic_dict["sum_of_cost"]), str(statistic_dict["time_step"]),
                                     str(sum(statistic_dict["penalty"])),str(statistic_dict["sic_final"]),str(statistic_dict["p"])+("({})".format(statistic_dict["f"]) if baseline_pscore else "")
                                     ))
        statistics.append(statistic_dict)


    count = 0
    sum_done_percent = 0
    sum_cost = 0
    num_done = 0
    sum_make=0
    sum_agents = 0
    sum_penalty = 0
    sum_sic_final = 0
    sum_p = None
    sum_f = 0
    sum_runtime = round(sum(runtimes),2)
    sum_ddl_met = 0
    for data in statistics:
        sum_done_percent += data["done_percentage"]
        sum_cost += data["sum_of_cost"]
        num_done += data["num_done"]
        sum_make += data["time_step"]
        sum_agents += data["No. of agents"]
        sum_penalty += sum(data["penalty"])
        sum_sic_final += data["sic_final"]
        sum_ddl_met += data["deadlines_met"]
        sum_f += data["f"]
        count+=1
    if question_type == 1:
        sum_p = int(sum_cost/sum_agents)
        pscores["q1"] = sum_p
        if baseline_pscore:
            sum_f = max(round(baseline_pscore["q1"]/sum_p,2),1.0)
    if save_pscore:
        with open(save_pscore,"w+") as f:
            f.write(json.dumps(pscores))
    print(output_template.format("Summary", str(sum_agents)+" (sum)", str(num_done)+" (sum)", str(sum_ddl_met)+"(sum)",str(sum_runtime)+"(sum)",
                                 str(sum_cost)+" (sum)", str(sum_make)+" (sum)", 
                                 str(sum_penalty)+" (sum)",str(sum_sic_final)+" (sum)",str(sum_p)+" (final)"+(str(sum_f) if baseline_pscore else "")
                                 ),flush=True)
    if write is not None:
        out.write(csv_template.format("Summary", str(sum_agents)+" (sum)", str(num_done)+" (sum)", str(sum_ddl_met)+"(sum)",str(sum_runtime)+"(sum)",
                                 str(sum_cost)+" (sum)", str(sum_make)+" (sum)", 
                                 str(sum_penalty)+" (sum)",str(sum_sic_final)+" (sum)",str(sum_p)+" (final)"+(str(sum_f) if baseline_pscore else "")
                                 ))
        out.close()
    if not mute:
        input("Press enter to exit:")

def remote_evaluator(get_path, args, replan=None):
    args = parser.parse_args(args[1:])
    path = args.tests
    q =args.q
    tests = glob.glob("{}/level_*/test_*.pkl".format(path))
    tests.sort()
    if q == 1:
        evaluator(get_path,tests,False,False,1,mute = True,write=args.o)
    elif q == 2:
        evaluator(get_path,tests,False,False,2,mute = True,write=args.o)
    elif q == 3:
        deadline_files =  [test.replace(".pkl",".ddl") for test in tests]
        evaluator(get_path,tests, False, False, 3, deadline_files,penalty_scale=4,mute = True, write=args.o, replan=replan)





