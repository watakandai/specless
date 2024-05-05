# To load MiniGrid-BlockedUnlockPickup-v0
import gym_minigrid  # noqa: F401
import gymnasium as gym

from specless.automaton.transition_system import MinigridTransitionSystem, TSBuilder
from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper


def test_transition_system_builder():
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    env = MiniGridTransitionSystemWrapper(env)

    tsbuilder = TSBuilder()
    transition_system = tsbuilder(env)
    assert isinstance(transition_system, MinigridTransitionSystem)


# TODO
def test_step():
    pass


# TODO:
def test_transition():
    pass


# TODO:
def test_run():
    pass
