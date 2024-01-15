# To load MiniGrid-BlockedUnlockPickup-v0
import gym_minigrid  # noqa: F401
import gymnasium as gym

import specless as sl
from specless.dataset import ArrayDataset
from specless.inference.partial_order import POInferenceAlgorithm
from specless.specification.base import Specification
from specless.specification.partial_order import PartialOrder
from specless.wrapper.utils import collect_demonstrations


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
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")

    # TODO: Define a function that maps a demo to a trace

    # Collect Demonstrations
    demonstrations = collect_demonstrations(env, num=10, only_success=True)

    # Convert them to a Dataset Class
    demonstrations = sl.ArrayDataset(demonstrations, columns=["symbol"])  # state labels

    inference = POInferenceAlgorithm()
    specification: Specification = inference.infer(demonstrations)

    assert isinstance(specification, PartialOrder)
