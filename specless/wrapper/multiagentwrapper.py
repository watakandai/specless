from enum import IntEnum
from typing import List

import gymnasium as gym

from specless.typing import StepData


class MultiAgentWrapper(gym.core.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def step(self, actions: List[IntEnum]) -> StepData:
        assert isinstance(actions, List)

        obs = {}
        reward: float = 0
        terminated: bool = False
        truncated: bool = False
        info = {}
        for action in actions:
            if action is None:
                continue
            o, r, te, tr, i = self.env.step(action)
            obs[action] = o
            reward += r
            terminated = truncated or te
            truncated = truncated or tr
            info[action] = i

        return obs, reward, terminated, truncated, info
