# To load MiniGrid-BlockedUnlockPickup-v0
import gym_minigrid  # noqa: F401
import gymnasium as gym

import specless as sl
from specless.dataset import ArrayDataset
from specless.inference.partial_order import POInferenceAlgorithm
from specless.specification.base import Specification
from specless.specification.partial_order import PartialOrder
from specless.utils.collect_demos import collect_demonstrations
from specless.wrapper.labelwrapper import LabelMiniGridWrapper
from specless.wrapper.selectstatewrapper import SelectStateDataWrapper


def test_construction():
    inference = POInferenceAlgorithm()
    assert True


def test_inference():
    inference = POInferenceAlgorithm()
    demonstrations: list = [
        ["a", "b", "c"],
        ["d", "e", "f"],
    ]
    columns: list = ["symbol"]
    trace_dataset = ArrayDataset(demonstrations, columns)
    specification: Specification = inference.infer(trace_dataset)
    assert isinstance(specification, PartialOrder)


def test_inference_on_gym_env():
    # TODO: Create a wrapper that only return the symbol and timestamp
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = LabelMiniGridWrapper(env, labelkey="label")
    env = SelectStateDataWrapper(env, columns=["label"])

    # Collect Demonstrations
    demonstrations = collect_demonstrations(
        env,
        only_success=False,
        add_timestamp=True,
        num=10,
        timeout=1000,
    )

    # Convert them to a Dataset Class
    demonstrations = sl.ArrayDataset(demonstrations, columns=["timestamp", "symbol"])

    inference = POInferenceAlgorithm()
    specification: Specification = inference.infer(demonstrations)
    assert isinstance(specification, PartialOrder)
