
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
from collections import deque


# import necessary modules that this python scripts need.
# The evaluation environment used in the assignment ships a module
# `flatland.utils.controller`.  The open source `flatland-rl` package
# does not include it, so during local testing the import may fail.  We
# therefore keep the import inside a ``try`` block to allow the file to
# be syntax‑checked even without the evaluation package installed.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import (
        get_action,
        Train_Actions,
        Directions,
        check_conflict,
        path_controller,
        evaluator,
        remote_evaluator,
    )
except Exception as e:  # pragma: no cover - handled in testing environment
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


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Simple Manhattan distance used for agent prioritisation."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



def _is_reserved(res_pos, res_edge, from_pos, to_pos, time) -> bool:
    """Check vertex and edge conflicts in the reservation tables."""
    if res_pos.get((to_pos[0], to_pos[1], time)):
        return True
    if res_edge.get((from_pos, to_pos, time)) or res_edge.get((to_pos, from_pos, time)):
        return True
    return False


def _reserve_path(res_pos, res_edge, path: List[Tuple[int, int]], start_time: int = 0) -> None:
    """Reserve cells and edges for a computed path from ``start_time``."""
    for i in range(len(path)):
        t = start_time + i
        pos = path[i]
        res_pos[(pos[0], pos[1], t)] = True
        if i > 0:
            res_edge[(path[i - 1], pos, t)] = True



def _search_single(rail: GridTransitionMap, start_pos: Tuple[int, int], start_dir: int,
                   target: Tuple[int, int], res_pos, res_edge, start_time: int,
                   max_timestep: int) -> List[Tuple[int, int]]:
    """Breadth first search in time-space avoiding existing reservations."""
    q = deque([(start_pos, start_dir, start_time, [start_pos])])
    visited = {(start_pos, start_dir, start_time)}

    while q:
        pos, direction, t, path = q.popleft()
        if pos == target:
            return path
        if t >= max_timestep - 1:
            continue

        next_time = t + 1

        # Option 1: wait in place
        if not _is_reserved(res_pos, res_edge, pos, pos, next_time):
            state = (pos, direction, next_time)
            if state not in visited:
                visited.add(state)
                q.append((pos, direction, next_time, path + [pos]))


        # Option 2: move along any valid transition
        valid_transitions = rail.get_transitions(pos[0], pos[1], direction)
        for nd in range(len(valid_transitions)):
            if not valid_transitions[nd]:
                continue
            nx, ny = pos
            if nd == Directions.NORTH:
                nx -= 1
            elif nd == Directions.EAST:
                ny += 1
            elif nd == Directions.SOUTH:
                nx += 1
            elif nd == Directions.WEST:
                ny -= 1
            new_pos = (nx, ny)
            if _is_reserved(res_pos, res_edge, pos, new_pos, next_time):
                continue

            state = (new_pos, nd, next_time)
            if state in visited:
                continue
            visited.add(state)
            q.append((new_pos, nd, next_time, path + [new_pos]))


    # No path found – remain in place
    return [start_pos]

# This function returns a list of location tuples as the solution.
# @param env The flatland railway environment
# @param agents A list of EnvAgent.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    # Reservation tables for vertices and edges
    res_pos = {}
    res_edge = {}

    n_agents = len(agents)
    paths = [None] * n_agents


    # Prioritise agents with shorter Manhattan distance to goal
    order = sorted(range(n_agents), key=lambda i: _manhattan(agents[i].initial_position, agents[i].target))

    for agent_id in order:
        agent = agents[agent_id]
        path = _search_single(

            rail,
            agent.initial_position,
            agent.initial_direction,
            agent.target,
            res_pos,
            res_edge,

            0,
            max_timestep,
        )

        # Extend the path by waiting at the goal to avoid later collisions
        if len(path) < max_timestep:
            path = path + [path[-1]] * (max_timestep - len(path))

        paths[agent_id] = path
        _reserve_path(res_pos, res_edge, path)


    return paths

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
    affected = set(new_malfunction_agents) | set(failed_agents)
    if not affected:
        return existing_paths

    # Build reservation tables from unaffected agents
    res_pos = {}
    res_edge = {}
    for idx, path in enumerate(existing_paths):
        if idx in affected:
            continue
        if current_timestep < len(path):
            _reserve_path(res_pos, res_edge, path[current_timestep:], current_timestep)

    new_paths = existing_paths[:]

    for idx in affected:
        agent = agents[idx]
        if len(existing_paths[idx]) > current_timestep:
            start = existing_paths[idx][current_timestep]
            if current_timestep > 0 and len(existing_paths[idx]) >= 2:
                prev = existing_paths[idx][current_timestep - 1]

                dx, dy = start[0] - prev[0], start[1] - prev[1]
                if dx == -1:
                    direction = Directions.NORTH
                elif dy == 1:
                    direction = Directions.EAST
                elif dx == 1:
                    direction = Directions.SOUTH
                elif dy == -1:
                    direction = Directions.WEST
                else:
                    direction = agent.initial_direction
            else:
                direction = agent.initial_direction

            prefix = existing_paths[idx][:current_timestep]
        else:
            # agent already finished path; restart from last position

            start = existing_paths[idx][-1]
            if len(existing_paths[idx]) >= 2:
                prev = existing_paths[idx][-2]
                dx, dy = start[0] - prev[0], start[1] - prev[1]
                if dx == -1:
                    direction = Directions.NORTH
                elif dy == 1:
                    direction = Directions.EAST
                elif dx == 1:
                    direction = Directions.SOUTH
                elif dy == -1:
                    direction = Directions.WEST
                else:
                    direction = agent.initial_direction
            else:
                direction = agent.initial_direction
            prefix = existing_paths[idx]


        replanned = _search_single(

            rail,
            start,
            direction,
            agent.target,
            res_pos,
            res_edge,

            current_timestep,
            max_timestep,
        )


        full_path = prefix + replanned[1:]
        if len(full_path) < max_timestep:
            full_path = full_path + [full_path[-1]] * (max_timestep - len(full_path))
        new_paths[idx] = full_path

        _reserve_path(res_pos, res_edge, full_path[current_timestep:], current_timestep)


    return new_paths


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




