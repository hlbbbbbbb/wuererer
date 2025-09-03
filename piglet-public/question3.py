from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json, random
from collections import deque

# import necessary modules that this python scripts need.
# The evaluation environment used in the assignment ships a module
# `flatland.utils.controller`.  The open source `flatland-rl` package
# does not include it, so during local testing the import may fail.  We
# therefore keep the import inside a ``try`` block to allow the file to
# be syntax‑checked even without the evaluation package installed.
try:  # pragma: no cover - handled in testing environment
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
except Exception as e:  # pragma: no cover
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
# Helper utilities
#########################

# Global caches reused between get_path and replan
_agent_deadlines: List[int] = []
_agent_edts: List[int] = []
_agent_speeds: List[int] = []
_agent_slacks: List[int] = []


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Simple Manhattan distance used for heuristics."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _get_deadline(agent: EnvAgent, default: int) -> int:
    """Return expected arrival time if available."""
    for attr in ("latest_arrival", "deadline", "expected_time"):
        if hasattr(agent, attr):
            return getattr(agent, attr)
    return default


def _get_edt(agent: EnvAgent) -> int:
    """Earliest departure time."""
    for attr in ("earliest_departure", "start_time", "earliest_start"):
        if hasattr(agent, attr):
            return getattr(agent, attr)
    return 0


def _get_speed(agent: EnvAgent) -> int:
    """Return discrete speed counter (Cmax)."""
    if hasattr(agent, "speed_data") and isinstance(agent.speed_data, dict):
        if "speed" in agent.speed_data and agent.speed_data["speed"] != 0:
            return int(round(1 / agent.speed_data["speed"]))
    if hasattr(agent, "speed") and agent.speed:
        return int(round(1 / agent.speed)) if agent.speed <= 1 else int(agent.speed)
    return 1


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


def _search_sipp(
    rail: GridTransitionMap,
    start_pos: Tuple[int, int],
    start_dir: int,
    target: Tuple[int, int],
    res_pos,
    res_edge,
    start_time: int,
    earliest_departure: int,
    speed_cmax: int,
    max_timestep: int,
):
    """SIPP-like search with discrete speeds and reservations."""

    t0 = max(start_time, earliest_departure)
    q = deque([(start_pos, start_dir, t0, 0, [start_pos])])
    visited = {(start_pos, start_dir, t0, 0)}

    while q:
        pos, direction, t, counter, path = q.popleft()
        if pos == target:
            return path
        if t >= max_timestep - 1:
            continue

        next_time = t + 1

        # Wait in place
        if not _is_reserved(res_pos, res_edge, pos, pos, next_time):
            new_counter = min(counter + 1, speed_cmax - 1)
            state = (pos, direction, next_time, new_counter)
            if state not in visited:
                visited.add(state)
                q.append((pos, direction, next_time, new_counter, path + [pos]))

        # Move if speed counter allows
        if counter + 1 >= speed_cmax:
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
                state = (new_pos, nd, next_time, 0)
                if state in visited:
                    continue
                visited.add(state)
                q.append((new_pos, nd, next_time, 0, path + [new_pos]))

    # No path found – remain in place
    return [start_pos]


def _arrival_time(path: List[Tuple[int, int]], goal: Tuple[int, int]) -> int:
    for t, p in enumerate(path):
        if p == goal:
            return t
    return len(path)


def _total_delay(paths: List[List[Tuple[int, int]]], agents: List[EnvAgent]) -> int:
    total = 0
    for i, path in enumerate(paths):
        deadline = _agent_deadlines[i]
        at = _arrival_time(path, agents[i].target)
        total += max(0, at - deadline)
    return total


def _lns_improve(paths, agents, rail, max_timestep, iterations=20):
    """Delay-based neighbourhood selection for LNS optimisation."""
    global _agent_slacks

    for _ in range(iterations):
        late = [i for i in range(len(agents)) if _arrival_time(paths[i], agents[i].target) > _agent_deadlines[i]]
        if not late:
            break
        seed = random.choice(late)
        neighbourhood = {seed}
        seed_path = paths[seed]
        for t, pos in enumerate(seed_path):
            for j, p in enumerate(paths):
                if j == seed:
                    continue
                if t < len(p) and p[t] == pos:
                    neighbourhood.add(j)
        subset = list(neighbourhood)

        res_pos, res_edge = {}, {}
        for idx, p in enumerate(paths):
            if idx in neighbourhood:
                continue
            _reserve_path(res_pos, res_edge, p)

        new_paths = paths[:]
        order = sorted(subset, key=lambda i: (_agent_slacks[i], _agent_speeds[i]))
        for idx in order:
            a = agents[idx]
            p = _search_sipp(
                rail,
                a.initial_position,
                a.initial_direction,
                a.target,
                res_pos,
                res_edge,
                0,
                _agent_edts[idx],
                _agent_speeds[idx],
                max_timestep,
            )
            if len(p) < max_timestep:
                p = p + [p[-1]] * (max_timestep - len(p))
            new_paths[idx] = p
            _reserve_path(res_pos, res_edge, p)

        if _total_delay(new_paths, agents) < _total_delay(paths, agents):
            paths = new_paths

    return paths


# ---------------------------------------------------------------------------
# Planning entry points
# ---------------------------------------------------------------------------


def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    global _agent_deadlines, _agent_edts, _agent_speeds, _agent_slacks

    n_agents = len(agents)
    _agent_deadlines = [_get_deadline(a, max_timestep) for a in agents]
    _agent_edts = [_get_edt(a) for a in agents]
    _agent_speeds = [_get_speed(a) for a in agents]
    _agent_slacks = [
        _agent_deadlines[i] - _agent_edts[i] - _manhattan(agents[i].initial_position, agents[i].target)
        for i in range(n_agents)
    ]

    res_pos, res_edge = {}, {}
    paths = [None] * n_agents

    order = sorted(range(n_agents), key=lambda i: (_agent_slacks[i], _agent_speeds[i]))

    for idx in order:
        a = agents[idx]
        path = _search_sipp(
            rail,
            a.initial_position,
            a.initial_direction,
            a.target,
            res_pos,
            res_edge,
            0,
            _agent_edts[idx],
            _agent_speeds[idx],
            max_timestep,
        )
        if len(path) < max_timestep:
            path = path + [path[-1]] * (max_timestep - len(path))
        paths[idx] = path
        _reserve_path(res_pos, res_edge, path)

    # Large neighbourhood search to reduce delays
    paths = _lns_improve(paths, agents, rail, max_timestep)

    return paths


def replan(
    agents: List[EnvAgent],
    rail: GridTransitionMap,
    current_timestep: int,
    existing_paths: List[List[Tuple[int, int]]],
    max_timestep: int,
    new_malfunction_agents: List[int],
    failed_agents: List[int],
):
    affected = set(new_malfunction_agents) | set(failed_agents)
    if not affected:
        return existing_paths

    res_pos, res_edge = {}, {}
    for idx, path in enumerate(existing_paths):
        if idx in affected:
            continue
        if current_timestep < len(path):
            _reserve_path(res_pos, res_edge, path[current_timestep:], current_timestep)

    new_paths = existing_paths[:]
    for idx in affected:
        agent = agents[idx]
        if current_timestep < len(existing_paths[idx]):
            start = existing_paths[idx][current_timestep]
            if current_timestep > 0:
                prev = existing_paths[idx][current_timestep - 1]
            else:
                prev = start
        else:
            start = existing_paths[idx][-1]
            prev = existing_paths[idx][-1]

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

        prefix = existing_paths[idx][:current_timestep]

        replanned = _search_sipp(
            rail,
            start,
            direction,
            agent.target,
            res_pos,
            res_edge,
            current_timestep,
            max(_agent_edts[idx], current_timestep),
            _agent_speeds[idx],
            max_timestep,
        )

        full_path = prefix + replanned[1:]
        if len(full_path) < max_timestep:
            full_path = full_path + [full_path[-1]] * (max_timestep - len(full_path))
        new_paths[idx] = full_path
        _reserve_path(res_pos, res_edge, full_path[current_timestep:], current_timestep)

    # Optional improvement pass on affected agents
    subset_agents = [agents[i] for i in range(len(agents))]
    new_paths = _lns_improve(new_paths, agents, rail, max_timestep, iterations=10)

    return new_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv, replan=replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(
                os.path.join(script_path, "multi_test_case/level{}_test_{}.pkl".format(level, test))
            )
        test_cases.sort()
        deadline_files = [test.replace(".pkl", ".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan=replan)

