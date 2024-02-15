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


def mainfunc(num_locations, solver_name, iteration):
    args = (num_locations, iteration)

    if args in ARGS_TO_TSP:
        env, tsp, tspbuilder = ARGS_TO_TSP[args]
    else:
        env = gym.make(
            ENV_ID,
            num_locations=5,
            width=20,
            height=20,
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

        ARGS_TO_TSP[args] = env, tsp, tspbuilder

    ##### Solve the TSP and obtain tours
    start_time: float = time.time()
    if solver_name == "milp":
        solver = sl.MILPTSPSolver()
    else:
        solver = sl.ORTSPSolver(
            first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
            timeout=60 * 10,
        )
    tours, cost = solver.solve(tsp)
    timetook = time.time() - start_time

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


if __name__ == "__main__":
    experiment_func: Callable[[Any], Tuple] = mainfunc
    arg_dict: Dict[str, List] = {
        "Node": [5, 10, 20],
        "Solver": ["milp", "ortools"],
        "Iteration": [1, 2, 3, 4, 5],
    }
    return_key_strs: List[str] = ["Tours", "Cost", "Time[s]"]
    csvfilepath: str = str(Path(__file__).with_suffix("").with_suffix(".csv"))
    logger = sl.BenchmarkLogger()
    logger.start(experiment_func, arg_dict, return_key_strs, csvfilepath)
