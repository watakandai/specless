"""
"""
from typing import Dict, List

import gymnasium as gym
from gym_minigrid.minigrid import MiniGridEnv


class SelectStateDataWrapper(gym.core.Wrapper):
    def __init__(self, env: MiniGridEnv, columns: List[str]):
        super().__init__(env)

        state, _ = env.reset()
        assert all([c in state for c in columns])
        self.columns = columns

    def _select_obs(self, obs) -> List:
        """Select desired observations from the full observations"""
        return [obs[c] for c in self.columns]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._select_obs(obs), info

    def step(self, action):
        """Step function that returns the selected observations"""
        obs: Dict
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._select_obs(obs)
        return obs, reward, terminated, truncated, info
