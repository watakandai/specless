import copy
from typing import List

import gym_minigrid  # noqa: F401 # To load the registered MiniGrid envs
import gymnasium as gym  # noqa
from gymnasium.core import ActType

from specless.automaton.transition_system import MinigridTransitionSystem, TSBuilder
from specless.factory.tspbuilder import TSPWithTPOBuilder
from specless.inference.timed_partial_order import TPOInferenceAlgorithm
from specless.specification.base import Specification
from specless.strategy import CombinedStrategy, PlanStrategy
from specless.synthesis import TSPSynthesisAlgorithm
from specless.tsp.solver.milp import MILPTSPWithTPOSolver
from specless.tsp.tsp import TSPWithTPO
from specless.utils.collect_demos import collect_demonstrations, simulate
from specless.wrapper.labelwrapper import LabelMiniGridWrapper
from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper
from specless.wrapper.selectstatewrapper import SelectStateDataWrapper



def test_tsp_synthesis():
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = LabelMiniGridWrapper(
        env, labelkey="label", skiplist=["unseen", "wall", "empty"]
    )

    ### Inference

    # Collect Data
    demonstrations = [
        [[00, "goal_green"]],
        [[20, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
    ]
    # Inference
    inference = TPOInferenceAlgorithm()
    tpo: Specification = inference.infer(demonstrations)

    ### Synthesis

    # Env -> TransitionSystem
    env = MiniGridTransitionSystemWrapper(env)
    tsbuilder = TSBuilder()
    transition_system: MinigridTransitionSystem = tsbuilder(env)
    # transition_system.draw("MiniGrid-Empty-5x5-v0")

    # TPO & TransitionSystem -> TSP
    tspbuilder = TSPWithTPOBuilder()
    tsp_with_tpo: TSPWithTPO = tspbuilder(transition_system, tpo)

    # Solve TSP -> Tours
    tspsolver = MILPTSPWithTPOSolver()
    tours, cost, timestamps = tspsolver.solve(tsp_with_tpo)

    # Convert tours to a sequence of actions...
    actions: List[ActType] = [tspbuilder.map_back_to_controls(tour) for tour in tours]

    if len(actions) == 0:
        assert False

    # Tours -> Strategy
    if len(actions) == 1:
        strategy = PlanStrategy(actions[0])
    else:
        strategy = CombinedStrategy([PlanStrategy(action) for action in actions])

    # Simulate
    simulate(env, strategy)

    assert True


def test_wrapped() -> None:
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = LabelMiniGridWrapper(
        env, labelkey="label", skiplist=["unseen", "wall", "empty"]
    )

    ### Inference

    # Collect Data
    demonstrations = [
        [[00, "goal_green"]],
        [[20, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
    ]

    # Inference
    inference = TPOInferenceAlgorithm()
    tpo: Specification = inference.infer(demonstrations)

    ### Synthesis

    # Synthesize
    algorithm = TSPSynthesisAlgorithm()
    env = MiniGridTransitionSystemWrapper(env, ignore_direction=True)
    strategy = algorithm.synthesize(env, tpo)
