"""

Label MiniGrid Wrapper

import gym_minigrid  # noqa: F401
import gymnasium as gym
from specless.env.labelwrapper import LabelMiniGridWrapper

env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
env = LabelMiniGridWrapper(env)
state, info = env.reset()
print(state)

label = env.get_label_from_state(state)
print(label)

Returns:
    _type_: _description_
"""

from typing import Dict

import gymnasium as gym
from gym_minigrid.minigrid import MiniGridEnv

from specless.const import IDX_TO_COLOR, IDX_TO_OBJECT, OBJECT_TO_IDX


class LabelMiniGridWrapper(gym.core.Wrapper):
    def __init__(self, env: MiniGridEnv, labelkey: str = "label", skiplist=[]) -> None:
        super().__init__(env)
        self.labelkey = labelkey
        self.skiplist = skiplist

    def get_label_from_state(self, state: Dict) -> str:
        """Get label from state

        Args:
            state (Dict): _description_

        Returns:
            _type_: _description_
        """
        base_env = self.env.unwrapped
        cell = base_env.grid.get(*base_env.agent_pos)
        if cell is None:
            obj_type = OBJECT_TO_IDX["empty"]
            obs = (obj_type, 0, 0)
        else:
            obs = cell.encode()

        obj_type, obj_color, obj_state = obs
        # Return '' if in the skiplist
        if IDX_TO_OBJECT[obj_type] in self.skiplist:
            return ""

        prop_string_base = "_".join([IDX_TO_OBJECT[obj_type], IDX_TO_COLOR[obj_color]])
        # return "_".join([prop_string_base, DIR_TO_STRING[obj_state]])
        return prop_string_base

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        label = self.get_label_from_state(obs)
        obs.update({self.labelkey: label})
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        label = self.get_label_from_state(obs)
        obs.update({self.labelkey: label})
        return obs, reward, terminated, truncated, info


class AddPosDirToMiniGridWrapper(gym.core.Wrapper):
    def __init__(
        self, env: MiniGridEnv, poslabel: str = "pos", dirlabel: str = "dir"
    ) -> None:
        super().__init__(env)
        self.poslabel: str = poslabel
        self.dirlabel: str = dirlabel

    def get_label_from_state(self, state: Dict) -> str:
        """Get label from state

        Args:
            state (Dict): _description_

        Returns:
            _type_: _description_
        """
        base_env = self.env.unwrapped
        pos = tuple(base_env.agent_pos)
        dir = base_env.agent_dir

        state.update({"pos": pos, "dir": dir})
        return state

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.get_label_from_state(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.get_label_from_state(obs)
        return obs, reward, terminated, truncated, info
