import gymnasium as gym  # noqa
import gym_minigrid  # noqa: F401 # To load MiniGrid-BlockedUnlockPickup-v0

import specless as sl
from specless.dataset import ArrayDataset
from specless.inference.timed_partial_order import TPOInferenceAlgorithm
from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.wrapper.labelwrapper import LabelMiniGridWrapper
from specless.wrapper.selectstatewrapper import SelectStateDataWrapper
from specless.wrapper.utils import collect_demonstrations


def test_construction():
    TPOInferenceAlgorithm()
    assert True


def test_inference():
    inference = TPOInferenceAlgorithm()
    demonstrations: list = [
        [[1, "a"], [2, "b"], [3, "c"]],
        [[4, "d"], [5, "e"], [6, "f"]],
    ]
    columns: list = ["timestamp", "symbol"]
    timedtrace_dataset = ArrayDataset(demonstrations, columns)
    specification: Specification = inference.infer(timedtrace_dataset)
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
        only_success=True,
        add_timestamp=True,
        num=10,
        timeout=1000,
    )

    # TODO: OR Define a DataCollector class to collect data given "symbol" and "timestamp" columns,
    # and returns a Dataset Class

    # Convert them to a Dataset Class
    dataset = sl.ArrayDataset(demonstrations, columns=["timestamp", "symbol"])

    inference = TPOInferenceAlgorithm()
    specification: Specification = inference.infer(dataset)

    assert isinstance(specification, TimedPartialOrder)
