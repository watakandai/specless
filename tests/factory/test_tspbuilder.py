# To load MiniGrid-BlockedUnlockPickup-v0
import gym_minigrid  # noqa: F401
import gymnasium as gym

from specless.automaton.transition_system import TSBuilder
from specless.factory.tspadapter import MiniGridSytemAndTSPAdapter
from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper


def test_tspbuilder():
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    env = MiniGridTransitionSystemWrapper(env)

    tsbuilder = TSBuilder()
    transition_system = tsbuilder(env, "minigrid")

    tspadapter = MiniGridSytemAndTSPAdapter()
    # TODO:
    # specification = Specification()
    # tsp = tspadapter(transition_system, specification)

    # assert isinstance(tsp, TSP)
