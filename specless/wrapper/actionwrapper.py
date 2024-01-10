from enum import IntEnum
from typing import Any, Dict, Tuple

import gym
import numpy as np

from specless.wrapper.tswrapper import Done, Reward, StepData


class EightOmniDirectionActions(IntEnum):
    # move in this direction on the grid
    north = 0
    south = 1
    east = 2
    west = 3
    northeast = 4
    northwest = 5
    southeast = 6
    southwest = 7
    stay = 8


EIGHT_ACTION_TO_POS_DELTA: Dict[IntEnum, Tuple[int, int]] = {
    EightOmniDirectionActions.north: (0, -1),
    EightOmniDirectionActions.south: (0, 1),
    EightOmniDirectionActions.east: (1, 0),
    EightOmniDirectionActions.west: (-1, 0),
    EightOmniDirectionActions.northeast: (1, -1),
    EightOmniDirectionActions.northwest: (-1, -1),
    EightOmniDirectionActions.southeast: (1, 1),
    EightOmniDirectionActions.southwest: (-1, 1),
    EightOmniDirectionActions.stay: (0, 0),
}


class FourOmniDirectionActions(IntEnum):
    # move in this direction on the grid
    north = 0
    south = 1
    east = 2
    west = 3


FOUR_ACTION_TO_POS_DELTA: Dict[IntEnum, Tuple[int, int]] = {
    FourOmniDirectionActions.north: (0, -1),
    FourOmniDirectionActions.south: (0, 1),
    FourOmniDirectionActions.east: (1, 0),
    FourOmniDirectionActions.west: (-1, 0),
}


class DiagOmniDirectionActions(IntEnum):
    # move in this direction on the grid
    northeast = 0
    northwest = 1
    southeast = 2
    southwest = 3


DIAG_ACTION_TO_POS_DELTA: Dict[IntEnum, Tuple[int, int]] = {
    DiagOmniDirectionActions.northeast: (1, -1),
    DiagOmniDirectionActions.northwest: (-1, -1),
    DiagOmniDirectionActions.southeast: (1, 1),
    DiagOmniDirectionActions.southwest: (-1, 1),
}


class OmniDirectionActionWrapper(gym.core.Wrapper):
    def __init__(
        self,
        env,
        Actions: IntEnum,
        action_to_pos_delta_map: Dict[IntEnum, Tuple[int, int]],
    ) -> None:
        super().__init__(env)
        # Override the unwrapped env's actions
        self.unwrapped.actions = Actions
        self.action_to_pos_delta_map = action_to_pos_delta_map
        # Override the unwrapped env's action space
        num_actions: int = len(self.unwrapped.actions)
        self.unwrapped.action_space = gym.spaces.Discrete(num_actions)

        def get_action_str(a) -> Any:
            return self.unwrapped.actions._member_names_[a]

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {
            get_action_str(action): action for action in self.unwrapped.actions
        }
        self.ACTION_ENUM_TO_STR = dict(
            zip(self.ACTION_STR_TO_ENUM.values(), self.ACTION_STR_TO_ENUM.keys())
        )

    def _step_function(self, action: IntEnum) -> Tuple[Done, Reward]:
        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        start_pos = base_env.agent_pos

        # a diagonal action is really just two simple actions :)
        pos_delta = self.action_to_pos_delta_map[action]

        # Get the contents of the new cell of the agent
        new_pos = tuple(np.add(start_pos, pos_delta))
        new_cell = base_env.grid.get(*new_pos)

        if new_cell is None or new_cell.can_overlap():
            base_env.agent_pos = new_pos
        if new_cell is not None and new_cell.type == "goal":
            done = True
            reward = base_env._reward()
        if new_cell is not None and new_cell.type == "lava":
            done = True

        return done, reward

    def step(self, action: IntEnum) -> StepData:
        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        base_env.step_count += 1

        done, reward = self._step_function(action)

        if base_env.step_count >= base_env.max_steps:
            done = True

        obs = base_env.gen_obs()

        return obs, reward, done, {}


class DirectionalActionWrapper(gym.core.Wrapper):
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, env) -> None:
        super().__init__(env)
        # Override the unwrapped env's actions
        self.unwrapped.actions = DirectionalActionWrapper.Actions
        # Override the unwrapped env's action space
        num_actions: int = len(self.unwrapped.actions)
        self.unwrapped.action_space = gym.spaces.Discrete(num_actions)

        def get_action_str(a) -> Any:
            return self.unwrapped.actions._member_names_[a]

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {
            get_action_str(action): action for action in self.unwrapped.actions
        }
        self.ACTION_ENUM_TO_STR = dict(
            zip(self.ACTION_STR_TO_ENUM.values(), self.ACTION_STR_TO_ENUM.keys())
        )

    def _step_function(self, action: IntEnum) -> Tuple[Done, Reward]:
        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        # Get the position in front of the agent
        fwd_pos = base_env.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = base_env.grid.get(*fwd_pos)

        # Rotate left
        if action == base_env.actions.left:
            base_env.agent_dir -= 1
            if base_env.agent_dir < 0:
                base_env.agent_dir += 4

        # Rotate right
        elif action == base_env.actions.right:
            base_env.agent_dir = (base_env.agent_dir + 1) % 4

        # Move forward
        elif action == base_env.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                base_env.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == "goal":
                done = True
                reward = base_env._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                done = True

        # Pick up an object
        elif action == base_env.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if base_env.carrying is None:
                    base_env.carrying = fwd_cell
                    base_env.carrying.cur_pos = np.array([-1, -1])
                    base_env.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == base_env.actions.drop:
            if not fwd_cell and base_env.carrying:
                base_env.grid.set(*fwd_pos, base_env.carrying)
                base_env.carrying.cur_pos = fwd_pos
                base_env.carrying = None

        # Toggle/activate an object
        elif action == base_env.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(base_env, fwd_pos)

        # Done action (not used by default)
        elif action == base_env.actions.done:
            pass

        else:
            assert False, "unknown action"

        return done, reward

    def step(self, action: IntEnum) -> StepData:
        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        base_env.step_count += 1

        done, reward = self._step_function(action)

        if base_env.step_count >= base_env.max_steps:
            done = True

        obs = base_env.gen_obs()

        return obs, reward, done, {}


if __name__ == "__main__":
    # TODO: Pick an env
    env = gym.make("")
    env = OmniDirectionActionWrapper(
        env, EightOmniDirectionActions, EIGHT_ACTION_TO_POS_DELTA
    )
    env = OmniDirectionActionWrapper(
        env, FourOmniDirectionActions, FOUR_ACTION_TO_POS_DELTA
    )
    env = OmniDirectionActionWrapper(
        env, DiagOmniDirectionActions, DIAG_ACTION_TO_POS_DELTA
    )

    env = DirectionalActionWrapper(env)
