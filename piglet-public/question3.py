
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
from collections import deque
import heapq

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


def _slack(agent: EnvAgent, max_timestep: int) -> int:
    """Compute slack = deadline - EDT - distance."""
    latest = agent.latest_arrival if agent.latest_arrival is not None else max_timestep
    return latest - agent.earliest_departure - _manhattan(agent.initial_position, agent.target)


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


def _search_single(
    rail: GridTransitionMap,
    start_pos: Tuple[int, int],
    start_dir: int,
    target: Tuple[int, int],
    res_pos,
    res_edge,
    start_time: int,
    time_limit: int,
) -> List[Tuple[int, int]]:
    """A* search in time-space avoiding reservations."""

    h0 = _manhattan(start_pos, target)
    open_list = []
    heapq.heappush(open_list, (start_time + h0, start_time, start_pos, start_dir, [start_pos]))
    visited = set()

    while open_list:
        f, g, pos, direction, path = heapq.heappop(open_list)
        state = (pos, direction, g)
        if state in visited:
            continue
        visited.add(state)

        if pos == target:
            return path
        if g >= time_limit - 1:
            continue

        next_time = g + 1

        # Option 1: wait in place
        if not _is_reserved(res_pos, res_edge, pos, pos, next_time):
            heapq.heappush(
                open_list,
                (
                    next_time + _manhattan(pos, target),
                    next_time,
                    pos,
                    direction,
                    path + [pos],
                ),
            )

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
            heapq.heappush(
                open_list,
                (
                    next_time + _manhattan(new_pos, target),
                    next_time,
                    new_pos,
                    nd,
                    path + [new_pos],
                ),
            )

    # No path found – remain in place
    return [start_pos]


def _plan_with_retry(
    rail: GridTransitionMap,
    start_pos: Tuple[int, int],
    start_dir: int,
    target: Tuple[int, int],
    res_pos,
    res_edge,
    start_time: int,
    initial_limit: int,
    max_timestep: int,
) -> List[Tuple[int, int]]:
    """Iteratively expand the search horizon until a path is found or ``max_timestep``."""

    limit = max(initial_limit, start_time + 1)
    while True:
        path = _search_single(
            rail, start_pos, start_dir, target, res_pos, res_edge, start_time, limit
        )
        if path[-1] == target or limit >= max_timestep:
            return path
        limit = min(max_timestep, limit + 20)


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

    # Prioritise agents using slack-based ordering
    order = sorted(
        range(n_agents),
        key=lambda i: (
            _slack(agents[i], max_timestep),
            agents[i].earliest_departure,
            _manhattan(agents[i].initial_position, agents[i].target),
        ),
    )

    for agent_id in order:
        agent = agents[agent_id]
        start_time = max(0, agent.earliest_departure)
        dist = _manhattan(agent.initial_position, agent.target)
        deadline = agent.latest_arrival if agent.latest_arrival is not None else max_timestep
        slack = deadline - start_time - dist
        time_limit = min(max_timestep, start_time + dist + max(slack, 0) + 20)
        path = _plan_with_retry(
            rail,
            agent.initial_position,
            agent.initial_direction,
            agent.target,
            res_pos,
            res_edge,
            start_time,
            time_limit,
            max_timestep,
        )

        full_path = [agent.initial_position] * start_time + path
        if len(full_path) < max_timestep:
            full_path = full_path + [full_path[-1]] * (max_timestep - len(full_path))

        paths[agent_id] = full_path
        _reserve_path(res_pos, res_edge, path, start_time)

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
    for idx in sorted(affected, key=lambda i: _slack(agents[i], max_timestep)):
        agent = agents[idx]
        start_time = max(current_timestep, agent.earliest_departure)
        if len(existing_paths[idx]) > start_time:
            start = existing_paths[idx][start_time]
            if start_time > 0:
                prev = existing_paths[idx][start_time - 1]
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
            prefix = existing_paths[idx][:start_time]
        else:
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

        dist = _manhattan(start, agent.target)
        deadline = agent.latest_arrival if agent.latest_arrival is not None else max_timestep
        slack = deadline - start_time - dist
        time_limit = min(max_timestep, start_time + dist + max(slack, 0) + 20)
        replanned = _plan_with_retry(
            rail,
            start,
            direction,
            agent.target,
            res_pos,
            res_edge,
            start_time,
            time_limit,
            max_timestep,
        )

        full_path = prefix + replanned[1:]
        if len(full_path) < max_timestep:
            full_path = full_path + [full_path[-1]] * (max_timestep - len(full_path))
        new_paths[idx] = full_path
        _reserve_path(res_pos, res_edge, full_path[start_time:], start_time)

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




