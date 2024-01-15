from collections.abc import Iterable
from typing import Dict, Tuple

# To load MiniGrid-BlockedUnlockPickup-v0
import gym_minigrid  # noqa: F401
import gymnasium as gym

from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper


def test_minigridwrapper():
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    env = MiniGridTransitionSystemWrapper(env)
    actions = env.actions()
    assert isinstance(actions, Iterable)


def test_extract_transition_system():
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    env = MiniGridTransitionSystemWrapper(env)

    state, info = env.reset()
    node: Tuple = env._get_node_from_state(state)

    assert isinstance(node, Tuple)

    action = list(env.actions())[0]

    (
        dest_state,
        reward,
        terminated,
        truncated,
        info,
    ) = env.make_transition(state, action)

    assert "image" in dest_state
    assert "dir" in dest_state
    assert "pos" in dest_state

    assert isinstance(dest_state, Dict)
    assert isinstance(reward, (float, int))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, Dict)
