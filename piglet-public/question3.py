
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys,time,json

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)




#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

#########################
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan frunction calls.
#########################


# This function return a list of location tuple as the solution.
# @param env The flatland railway environment
# @param agents A list of EnvAgent.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(agents: List[EnvAgent],rail: GridTransitionMap, max_timestep: int):
    ############
    # Below is an dummy path finding implementation,
    # which always choose the first available transition of current state.
    #
    # Replace these with your implementation and return a list of paths. Each path is a list of (x,y) tuple as your plan.
    # Your plan should avoid conflicts with each other.
    ############

    # initialize path list
    path_all = []

    # for each agent in env
    for agent_id in range(0,len(agents)):
        path = []
        loc = agents[agent_id].initial_position
        direction = agents[agent_id].initial_direction


        for t in range(0, int(max_timestep/10)):
            # add loc to path list
            path.append(loc)
            if loc == agents[agent_id].target:
                break

            # get available transitions from Rail_Env object.
            valid_transitions = rail.get_transitions(loc[0],loc[1],direction)
            for i in range(0,len(valid_transitions)):
                if valid_transitions[i]:
                    new_x=loc[0]
                    new_y=loc[1]
                    action = i
                    if action == Directions.NORTH:
                        new_x -= 1
                    elif action == Directions.EAST:
                        new_y += 1
                    elif action == Directions.SOUTH:
                        new_x += 1
                    elif action == Directions.WEST:
                        new_y -= 1

                    conflict = False
                    for p in path_all:
                        if t+1 < len(p) and p[t+1] == (new_x,new_y):
                            conflict = True
                        if t+1 < len(p) and p[t+1] ==(loc[0],loc[1]) and  p[t] ==(new_x,new_y):
                            conflict = True
                    if conflict:
                        continue

                    loc = (new_x,new_y)
                    direction = action
                    break
        path_all.append(path)

    return path_all

# This function return a list of location tuple as the solution.
# @param rail The flatland railway GridTransitionMap
# @param agents A list of EnvAgent.
# @param current_timestep The timestep that malfunction/collision happens .
# @param existing_paths The existing paths from previous get_plan or replan.
# @param max_timestep The max timestep of this episode.
# @param new_malfunction_agents  The id of agents have new malfunction happened at current time step (Does not include agents already have malfunciton in past timesteps)
# @param failed_agents  The id of agents failed to reach the location on its path at current timestep.
# @return path_all  Return paths that locaitons from current_timestp is updated to handle malfunctions and failed execuations.
def replan(agents: List[EnvAgent],rail: GridTransitionMap,  current_timestep: int, existing_paths: List[Tuple], max_timestep:int, new_malfunction_agents: List[int], failed_agents: List[int]):
    if debug:
        print("Replan function not implemented yet!",file=sys.stderr)
    return existing_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv, replan = replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan = replan)




