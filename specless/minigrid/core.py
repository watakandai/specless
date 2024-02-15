import copy
from enum import IntEnum
from typing import Tuple

import gymnasium as gym
import numpy as np
from gym_minigrid.minigrid import Grid, MiniGridEnv
from gym_minigrid.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
)

from specless.typing import ActionsEnum, Done, EnvType, Reward, StepData

TILE_PIXELS = 128


class NoDirectionAgentGrid(Grid):
    """
    This class overrides the drawing of direction-less agents
    """

    tile_cache = {}

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def render(self, tile_size, agent_pos=None, agent_dir=None, highlight_mask=None):
        """
        Render this grid at a given scale

        NOTE: overridden here to change the tile rendering to be the class' own

        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))

                # CHANGED: Grid.render_tile(...) to self.render_tile(...)
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
        white_background=True,
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        if white_background:
            img = np.full(
                shape=(tile_size * subdivs, tile_size * subdivs, 3),
                fill_value=WHITE,
                dtype=np.uint8,
            )
        else:
            img = np.zeros(
                shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
            )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            cir_fn = point_in_circle(cx=0.5, cy=0.5, r=0.3)
            fill_coords(img, cir_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img


# Enumeration of possible actions
# as this is a static environment, we will only allow for movement actions
# For a simple environment, we only allow the agent to move:
# North, South, East, or West
class SimpleStaticActions(IntEnum):
    # move in this direction on the grid
    north = 0
    south = 1
    east = 2
    west = 3


SIMPLE_ACTION_TO_DIR_IDX = {
    SimpleStaticActions.north: 3,
    SimpleStaticActions.south: 1,
    SimpleStaticActions.east: 0,
    SimpleStaticActions.west: 2,
}


# Enumeration of possible actions
# as this is a static environment, we will only allow for movement actions
# For a simple environment, we only allow the agent to move:
# Northeast, Northwest, Southeast, or Southwest
class DiagStaticActions(IntEnum):
    # move in this direction on the grid
    northeast = 0
    northwest = 1
    southeast = 2
    southwest = 3


DIAG_ACTION_TO_POS_DELTA = {
    DiagStaticActions.northeast: (1, -1),
    DiagStaticActions.northwest: (-1, -1),
    DiagStaticActions.southeast: (1, 1),
    DiagStaticActions.southwest: (-1, 1),
}


# Enumeration of possible actions
# as this is a static environment, we will only allow for movement actions
class StaticActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class ModifyActionsWrapper(gym.core.Wrapper):
    """
    This class allows you to modify the action space and behavior of the agent

    :param      env:           The gym environment to wrap
    :param      actions_type:  The actions type string
                               {'static', 'simple_static', 'diag_static',
                                'default'}
                               'static':
                               use a directional agent only capable of going
                               forward and turning
                               'simple_static':
                               use a non-directional agent which can only move
                               in cardinal directions in the grid
                               'default':
                               use an agent which has the default MinigridEnv
                               actions, suitable for dynamic environments.
    """

    def __init__(self, env: EnvType, actions_type: str = "static"):
        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(env)

        self._allowed_actions_types = set(
            ["static", "simple_static", "diag_static", "default"]
        )
        if actions_type not in self._allowed_actions_types:
            msg = f"actions_type ({actions_type}) must be one of: " + f"{actions_type}"
            raise ValueError(msg)

        # Need to change the Action enumeration in the base environment.
        # This also changes the "step" behavior, so we also change that out
        # to match the new set of actions
        self._actions_type = actions_type

        if actions_type == "static":
            actions = StaticActions
            step_function = self._step_default
        elif actions_type == "simple_static":
            actions = SimpleStaticActions
            step_function = self._step_simple_static
        elif actions_type == "diag_static":
            actions = DiagStaticActions
            step_function = self._step_diag_static
        elif actions_type == "default":
            actions = MiniGridEnv.Actions
            step_function = self._step_default

        self.unwrapped.actions = actions
        self._step_function = step_function

        # Actions are discrete integer values
        num_actions = len(self.unwrapped.actions)
        self.unwrapped.action_space = gym.spaces.Discrete(num_actions)

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {
            self._get_action_str(action): action for action in self.unwrapped.actions
        }
        self.ACTION_ENUM_TO_STR = dict(
            zip(self.ACTION_STR_TO_ENUM.values(), self.ACTION_STR_TO_ENUM.keys())
        )

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

    def _step_diag_static(self, action: IntEnum) -> Tuple[Done, Reward]:
        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        start_pos = base_env.agent_pos

        # a diagonal action is really just two simple actions :)
        pos_delta = DIAG_ACTION_TO_POS_DELTA[action]

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

    def _step_simple_static(self, action: IntEnum) -> Tuple[Done, Reward]:
        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        reward = 0
        done = False

        # save the original direction so we can reset it after moving
        old_dir = base_env.agent_dir
        new_dir = SIMPLE_ACTION_TO_DIR_IDX[action]
        base_env.agent_dir = new_dir

        # Get the contents of the cell in front of the agent
        fwd_pos = base_env.front_pos
        fwd_cell = base_env.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            base_env.agent_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == "goal":
            done = True
            reward = base_env._reward()
        if fwd_cell is not None and fwd_cell.type == "lava":
            done = True

        # reset the direction of the agent, as it really cannot change
        # direction
        base_env.agent_dir = old_dir

        return done, reward

    def _step_default(self, action: IntEnum) -> Tuple[Done, Reward]:
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

    def _get_action_str(self, action_enum: ActionsEnum) -> str:
        """
        Gets a string representation of the action enum constant

        :param      action_enum:  The action enum constant to convert

        :returns:   The action enum's string representation
        """

        return self.unwrapped.actions._member_names_[action_enum]


class MultiAgentWrapperEnv(gym.Env):
    def __init__(self, env, num_agent):
        self.envs = [copy.deepcopy(env) for i in range(num_agent)]
