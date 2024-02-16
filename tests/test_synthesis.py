import copy
from typing import List

import gym_minigrid  # noqa: F401 # To load the registered MiniGrid envs
import gymnasium as gym  # noqa
from gymnasium.core import ActType

from specless.automaton.transition_system import MinigridTransitionSystem, TSBuilder
from specless.dataset import ArrayDataset
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


def build_dataset_from_env(env_):
    env = copy.deepcopy(env_)
    env = SelectStateDataWrapper(env, columns=["label"])
    # Collect Demonstrations
    demonstrations = collect_demonstrations(
        env,
        only_success=True,
        add_timestamp=True,
        num=10,
        timeout=1000,
    )

    assert len(demonstrations) == 10
    dataset = ArrayDataset(demonstrations, columns=["timestamp", "symbol"])
    return dataset


def test_tsp_synthesis():
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = LabelMiniGridWrapper(
        env, labelkey="label", skiplist=["unseen", "wall", "empty"]
    )

    ### Inference

    # Collect Data
    # dataset: ArrayDataset = build_dataset_from_env(env)
    demonstrations = [
        [[00, "goal_green"]],
        [[20, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
    ]
    dataset = ArrayDataset(demonstrations, columns=["timestamp", "symbol"])
    # Inference
    inference = TPOInferenceAlgorithm()
    tpo: Specification = inference.infer(dataset)

    ### Synthesis

    # Env -> TransitionSystem
    env = MiniGridTransitionSystemWrapper(env)
    tsbuilder = TSBuilder()
    transition_system: MinigridTransitionSystem = tsbuilder(
        env, graph_data_format="minigrid"
    )
    # transition_system.draw("MiniGrid-Empty-5x5-v0")

    # TPO & TransitionSystem -> TSP
    tspbuilder = TSPWithTPOBuilder()
    tsp_with_tpo: TSPWithTPO = tspbuilder(transition_system, tpo)

    # Solve TSP -> Tours
    tspsolver = MILPTSPWithTPOSolver()
    tours, cost = tspsolver.solve(tsp_with_tpo)

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


def test_wrapped():
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = LabelMiniGridWrapper(
        env, labelkey="label", skiplist=["unseen", "wall", "empty"]
    )

    ### Inference

    # Collect Data
    # dataset: ArrayDataset = build_dataset_from_env(env)
    demonstrations = [
        [[00, "goal_green"]],
        [[20, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
        # [[20, "empty_red"], [50, "goal_green"]],
    ]
    dataset = ArrayDataset(demonstrations, columns=["timestamp", "symbol"])

    # Inference
    inference = TPOInferenceAlgorithm()
    tpo: Specification = inference.infer(dataset)

    ### Synthesis

    # Synthesize
    algorithm = TSPSynthesisAlgorithm()
    strategy = algorithm.synthesize(tpo, env)
