# To load MiniGrid-BlockedUnlockPickup-v0
import gym_minigrid  # noqa: F401
import gymnasium as gym

gym_minigrid.register_minigrid_envs()

from specless.automaton.transition_system import TSBuilder
from specless.factory.tspadapter import MiniGridSytemAndTSPAdapter
from specless.tsp.tsp import TSP
from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper


def test_tspbuilder():
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    env = MiniGridTransitionSystemWrapper(env)

    tsbuilder = TSBuilder()
    transition_system = tsbuilder(env, "minigrid")

    tspadapter = MiniGridSytemAndTSPAdapter()
    tsp = tspadapter(transition_system)

    assert isinstance(tsp, TSP)
