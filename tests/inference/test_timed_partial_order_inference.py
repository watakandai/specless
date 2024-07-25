import gymnasium as gym  # noqa
import gym_minigrid  # noqa: F401 # To load MiniGrid-BlockedUnlockPickup-v0

import specless as sl
from specless.inference.timed_partial_order import TPOInferenceAlgorithm
from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.wrapper.labelwrapper import LabelMiniGridWrapper
from specless.wrapper.selectstatewrapper import SelectStateDataWrapper
from specless.utils.collect_demos import collect_demonstrations


def test_construction():
    TPOInferenceAlgorithm()
    assert True


def test_inference():
    inference = TPOInferenceAlgorithm()
    demonstrations: list = [
        [[1, "a"], [2, "b"], [3, "c"]],
        [[4, "d"], [5, "e"], [6, "f"]],
    ]
    specification: Specification = inference.infer(demonstrations)
    assert isinstance(specification, TimedPartialOrder)


def test_inference_on_gym_env():
    # TODO: Create an env that collects a position, symbol, and timestamp.

    # TODO: Create a wrapper that only return the symbol and timestamp
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = LabelMiniGridWrapper(env, labelkey="label")
    env = SelectStateDataWrapper(env, columns=["label"])

    # Collect Demonstrations
    demonstrations = collect_demonstrations(
        env,
        only_finished=True,
        add_timestamp=True,
        num=10,
        timeout=1000,
    )

    inference = TPOInferenceAlgorithm()
    specification: Specification = inference.infer(demonstrations)

    assert isinstance(specification, TimedPartialOrder)
