"""

TODO:
- Change the Env direction easily
- Change the Env to multi agents
- Strategy -> Combined Strategy
-
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import random
import time
from typing import Any, Callable, Dict, List, Tuple

import gym_minigrid  # noqa: F401
import gymnasium as gym
from gymnasium.core import ActType
from ortools.constraint_solver import routing_enums_pb2

import specless as sl
from specless.minigrid.tspenv import TSPBenchmarkEnv  # NOQA
from specless.typing import EnvType

GYM_MONITOR_LOG_DIR = Path.cwd().joinpath(".gymlog")
ENV_ID = "MiniGrid-TSPBenchmarkEnv-v0"
ARGS_TO_TSP: Dict[Tuple, EnvType] = {}


def mainfunc(num_locations, num_agent, num_constraint_ratio, max_time_gap, solver_name, iteration):
    args = (num_locations, num_constraint_ratio, max_time_gap, iteration)

    if args in ARGS_TO_TSP:
        env, tsp_with_tpo, tspbuilder = ARGS_TO_TSP[args]
    else:
        env = gym.make(
            ENV_ID,
            num_locations=num_locations,
            width=30,
            height=30,
            agent_start_pos=(1, 5),
        )

        env = sl.MiniGridTransitionSystemWrapper(env, ignore_direction=True)

        ##### Convert a Transition System from an OpenAI Gym Environment (env)

        tsbuilder = sl.TSBuilder()
        transition_system = tsbuilder(env, "minigrid")

        ##### Convert the Transition System to a Traveling Saleseman Problem
        # Option 1: Ignore Directions
        # Option 2: Unique Label OR Treat them differently
        tspbuilder = sl.TSPBuilder()
        tsp: sl.TSP = tspbuilder(transition_system, uniquelabel=False)

        def random_time_gap():
            return random.randint(1, max_time_gap)

        initial_nodes = [tsp.nodes[0]]
        tpo = sl.generate_random_constraints(
            tsp.nodes,
            initial_nodes,
            tsp.costs,
            num_constraint_ratio,
            time_gap_callback=random_time_gap,
        )
        tsp_with_tpo = sl.TSPWithTPO.from_tsp(tsp, tpo)

        ARGS_TO_TSP[args] = env, tsp_with_tpo, tspbuilder

    ##### Solve the TSP and obtain tours
    start_time: float = time.time()
    if solver_name == "milp":
        solver = sl.MILPTSPWithTPOSolver()
        tours, cost, timestamps = solver.solve(tsp_with_tpo, num_agent=num_agent)
    else:
        solver = sl.ORTSPWithTPOSolver(
            first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
            timeout=60 * 10,
        )
        tours, cost = solver.solve(tsp_with_tpo, num_agent=num_agent)

    timetook = time.time() - start_time

    n = len(tours)
    for i in range(n):
        print(f"Tour{i}={tours[i]}, \tTimestamps={timestamps[i]}")

    analyzer = sl.RobustAnalysis()
    bound, times, finaltime = analyzer.analyze(tsp_with_tpo, tsp_with_tpo.tpo, tours)

    return tours, cost, timetook, bound
    """
    ##### Map the tours back onto the OpenAI Gym Environment to obtain a controller(s) (a sequence of actions)
    # TODO: Inlucd the actions -> strategy conversions into the tspbuilder
    actions: List[ActType] = [tspbuilder.map_back_to_controls(tour) for tour in tours]
    if len(actions) == 0:
        assert False
    ##### Convert a sequences of actions to a strategy class.
    if len(actions) == 1:
        strategy = sl.PlanStrategy(actions[0])
    else:
        strategy = sl.CombinedStrategy([sl.PlanStrategy(action) for action in actions])

    env = sl.TerminateIfNoStrategyWrapper(env)

    # TODO: Simply change it to the following using MultiAgentWrapper()
    # states, actions = sl.simulate(env, strategy)
    # print(states, actions)

    # TODO: Implement MultiAgentWrapper
    # This should hold multiple envs.
    if isinstance(strategy, sl.CombinedStrategy):
        for i, s in enumerate(strategy.strategies):
            states, actions = sl.simulate(env, s)
            # states = [(s["pos"], s["dir"], s["observation"]) for s in states]
            # print(f"Agent {i+1}: ")
            # print("\t\n".join(map(str, states)))
    else:
        states, actions = sl.simulate(env, strategy)

    return tours, cost, timetook
    """


if __name__ == "__main__":
    experiment_func: Callable[[Any], Tuple] = mainfunc
    arg_dict: Dict[str, List] = {
        "Node": [5, 10, 20, 40, 60, 80],
        # "Node": [100, 120, 140, 160],
        # "Agent": [1, 10, 20, 30, 40],
        "Agent": [2, 10, 20, 30, 40],
        "ConstraintRatio": [1.0, 0.75, 0.5, 0.25],
        "MaxTimeGap": [10, 30, 50],
        # "Solver": ["milp", "ortools"],
        "Solver": ["milp"],
        "Iteration": [1],
    }
    return_key_strs: List[str] = ["Tours", "Cost", "Time[s]", "Bound"]
    csvfilepath: str = str(Path(__file__).with_suffix("").with_suffix(".csv"))
    logger = sl.BenchmarkLogger()
    logger.start(experiment_func, arg_dict, return_key_strs, csvfilepath)
