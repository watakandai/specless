import random
from enum import IntEnum
from typing import Callable, Dict, List

import distinctipy
import gym
import numpy as np
from gym_minigrid.minigrid import (
    COLORS,
    Box,
    Floor,
    Grid,
    Lava,
    MiniGridEnv,
    fill_coords,
    point_in_rect,
)
from gymnasium import register
from wombats.systems.minigrid import (
    MINIGRID_TO_GRAPHVIZ_COLOR,
    Agent,
    MultiAgentMiniGridEnv,
    MultiObjGrid,
    NoDirectionAgentGrid,
    StepData,
)


class TSPEnv(MiniGridEnv):
    """TSP Environment with Multiple floor locations with duplicate colors"""

    def __init__(
        self,
        width=20,
        height=20,
        agent_start_pos=(1, 5),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.directionless_agent = True

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        lava_locs = [
            (15, 5),
            (3, 1),
            (16, 6),
            (3, 2),
            (2, 8),
            (1, 14),
            (3, 7),
            (14, 13),
            (8, 13),
            (14, 2),
        ]
        floor_locs = [
            (13, 7),
            (16, 12),
            (16, 5),
            (17, 11),
            (18, 10),
            (18, 15),
            (3, 15),
            (5, 18),
            (10, 10),
            (1, 15),
            (18, 18),
            (5, 5),
            (8, 3),
            (4, 1),
        ]

        for lava_loc in lava_locs:
            self.put_obj(Lava(), *lava_loc)

        # available_locs = [(i, j) for (i, j) in locations \
        #                          if (i,j) not in lava_locs]
        # floor_locs = random.sample(available_locs, len(MINIGRID_TO_GRAPHVIZ_COLOR))
        # print(floor_locs)
        N = len(MINIGRID_TO_GRAPHVIZ_COLOR)
        for i, loc in enumerate(floor_locs):
            ind = i % N
            color = list(MINIGRID_TO_GRAPHVIZ_COLOR.keys())[ind]
            self.put_obj(Floor(color=color), *loc)

        # available_locs = [(i, j) for (i, j) in available_locs \
        #                          if (i,j) not in floor_locs]
        # goal_locs = random.sample(available_locs, len(MINIGRID_TO_GRAPHVIZ_COLOR))
        # for color, loc in zip(MINIGRID_TO_GRAPHVIZ_COLOR, goal_locs):
        #     self.put_obj(Goal(color=color), *loc)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Get to a green and a purple tiles"


class MyFloor(Floor):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, rgb):
        color = "_".join([str(v) for v in rgb])
        super().__init__(color)
        self.rgb = rgb

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.rgb)


class TSPBenchmarkEnv(MiniGridEnv):
    """TSP Environment with Multiple floor locations with duplicate colors"""

    def __init__(
        self,
        num_locations=5,
        width=20,
        height=20,
        agent_start_pos=(1, 5),
        agent_start_dir=0,
    ):
        rgbs = distinctipy.get_colors(num_locations)
        f = lambda v: int(255 * v)
        rgbs = [tuple(map(f, rgb)) for rgb in rgbs]
        # N = len(gym_minigrid.minigrid.COLOR_TO_IDX)
        # COLOR_TO_IDX = {"_".join([str(v) for v in rgb]): N+i for i, rgb in enumerate(rgbs)}
        # gym_minigrid.minigrid.COLOR_TO_IDX.update(COLOR_TO_IDX)
        # gym_minigrid.minigrid.IDX_TO_COLOR = dict(zip(gym_minigrid.minigrid.COLOR_TO_IDX.values(), gym_minigrid.minigrid.COLOR_TO_IDX.keys()))
        # wombats.systems.minigrid.COLOR_TO_IDX = gym_minigrid.minigrid.COLOR_TO_IDX
        # wombats.systems.minigrid.IDX_TO_COLOR = gym_minigrid.minigrid.COLOR_TO_IDX
        self.rgbs = rgbs
        self.locations = self.get_random_locations(num_locations, width, height)
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.directionless_agent = True

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            see_through_walls=True,
        )

    def get_random_locations(self, N, width, height):
        coordinates = set()
        while len(coordinates) < N:
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            coordinate = (x, y)
            coordinates.add(coordinate)
        return list(coordinates)

    def _gen_grid(self, width, height):
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        f = lambda v: int(255 * v)
        for i, loc in enumerate(self.locations):
            rgb = self.rgbs[i]
            self.put_obj(Floor("red"), *loc)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Get to a green and a purple tiles"


class MyBox(Box):
    def __init__(self, color, contains=None, toggletimes=1, triage_color=None):
        super().__init__(color, contains, toggletimes, triage_color)

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))


class AircraftTurnaroundEnv(MiniGridEnv):
    """Aircraft Turnaround Environment
    where multiple agents must plan to complete tasks
    in coorperation.
    This environment only allows one single agent.
    """

    def __init__(
        self,
        width=13,
        height=23,
        agent_start_pos=(1, 5),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.directionless_agent = True

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        lava_locs = [
            # Body
            # (6, 5),
            # (5, 5), (6, 5), (7, 5), (5, 6), (7, 6)
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),  # (6, 16), (6, 17),
            # Wing
            (7, 9),
            (8, 10),
            (9, 11),
            (7, 10),
            (8, 11),  # (10, 12),
            (5, 9),
            (4, 10),
            (3, 11),
            (5, 10),
            (4, 11),  # (2, 12),
            # Tail
            (7, 14),
            (8, 15),
            (7, 15),
            (5, 14),
            (4, 15),
            (5, 15),
        ]
        for lava_loc in lava_locs:
            self.put_obj(Lava(), *lava_loc)

        # Stairs Location
        self.put_obj(MyBox("red"), 5, 7)
        # Stairs Station
        self.put_obj(Floor("red"), 1, 7)

        # Loading Location
        self.put_obj(MyBox("green"), 7, 8)
        # Loading Station
        self.put_obj(Floor("green"), 11, 8)

        # Refueling Location
        self.put_obj(MyBox("blue"), 8, 9)
        # Refueling Station
        self.put_obj(Floor("blue"), 11, 9)

        # Catering & Cleaning Location
        self.put_obj(MyBox("purple"), 7, 12)
        # Catering & Cleaning Station
        self.put_obj(Floor("purple"), 10, 19)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Visit All Areas"


class AspenAircraftTurnaroundEnv(MiniGridEnv):
    """Aircraft Turnaround Environment
    where multiple agents must plan to complete tasks
    in coorperation.
    This environment only allows one single agent.
    """

    def __init__(
        self,
        width=13,
        height=23,
        agent_start_pos=(1, 5),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.directionless_agent = True

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        lava_locs = [
            # Body
            # (6, 5),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            # Wing
            (7, 9),
            (8, 10),
            (7, 10),
            (8, 11),
            (9, 11),  # (8, 9),
            (5, 9),
            (4, 10),
            (5, 10),
            (4, 11),
            (3, 11),  # (4, 9),
            # Tail
            (7, 14),
            (8, 15),
            (7, 15),  # (8, 14),
            (5, 14),
            (4, 15),
            (5, 15),  # (4, 14),
        ]
        for lava_loc in lava_locs:
            self.put_obj(Lava(), *lava_loc)

        # Stairs Location
        self.put_obj(MyBox("red"), 5, 7)
        # Stairs Station
        self.put_obj(Floor("red"), 1, 7)

        # Refueling Location
        self.put_obj(MyBox("blue"), 7, 7)
        # Refueling Station
        self.put_obj(Floor("blue"), 11, 7)

        # Loading Location
        self.put_obj(MyBox("green"), 7, 13)
        # Loading Station
        self.put_obj(Floor("green"), 11, 13)

        # Catering & Cleaning Location
        self.put_obj(MyBox("purple"), 5, 13)
        # Catering & Cleaning Station
        self.put_obj(Floor("purple"), 1, 20)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Visit All Areas"


class TSPMultiFourGrids(MultiAgentMiniGridEnv):
    """Test Env with just four grids"""

    def __init__(
        self,
        width=6,
        height=3,
        agent_start_pos_list=[(1, 1), (4, 1)],
        agent_start_dir_list=[0, 0],
        agent_colors=["red", "blue"],
        directionless_agent=True,
    ):
        self.agent_start_pos_list = agent_start_pos_list
        self.agent_start_dir_list = agent_start_dir_list
        self.agent_colors = agent_colors
        self.goal_1_pos = (width - 2, 1)

        self.directionless_agent = directionless_agent

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True,
            concurrent=True,
        )

    def _gen_grid(self, width, height):
        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color="green"), *self.goal_1_pos)

        n_agent = len(self.agent_start_pos_list)

        # TODO: Place the agent
        for i in range(n_agent):
            p = self.agent_start_pos_list[i]
            d = self.agent_start_dir_list[i]
            c = self.agent_colors[i]
            if p is not None:
                self.put_agent(
                    Agent(name=f"agent{i}", color=c, view_size=self.view_size),
                    *p,
                    d,
                    True,
                )
            else:
                self.place_agent()

        self.mission = "get to the green squares"


# TODO: All you have to do is to define actions & step_funciton
class EightDirectionActionWrapper(gym.core.Wrapper):
    class EightDirectionActions(IntEnum):
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

    ACTION_TO_POS_DELTA = {
        EightDirectionActions.north: (0, -1),
        EightDirectionActions.south: (0, 1),
        EightDirectionActions.east: (1, 0),
        EightDirectionActions.west: (-1, 0),
        EightDirectionActions.northeast: (1, -1),
        EightDirectionActions.northwest: (-1, -1),
        EightDirectionActions.southeast: (1, 1),
        EightDirectionActions.southwest: (-1, 1),
        EightDirectionActions.stay: (0, 0),
    }

    def __init__(self, env):
        super().__init__(env)
        self.unwrapped.actions = EightDirectionActionWrapper.EightDirectionActions
        num_actions = len(self.unwrapped.actions)
        self.unwrapped.action_space = gym.spaces.Discrete(num_actions)
        get_action_str = lambda a: self.unwrapped.actions._member_names_[a]

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {
            get_action_str(action): action for action in self.unwrapped.actions
        }
        self.ACTION_ENUM_TO_STR = dict(
            zip(self.ACTION_STR_TO_ENUM.values(), self.ACTION_STR_TO_ENUM.keys())
        )

    def _step_function(self, action: IntEnum):
        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        start_pos = base_env.agent_pos

        # a diagonal action is really just two simple actions :)
        pos_delta = EightDirectionActionWrapper.ACTION_TO_POS_DELTA[action]

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


class SingleToTurnBasedMultiAgentWrapper(gym.core.Wrapper):
    """Single to Turn-based MultiAgent Wrapper"""

    agents: List[int]
    agent_selector: Callable[..., int]
    curr_agent: int

    def __init__(self, env, num_agent: int):
        super().__init__(env)
        self.agents = list(range(num_agent))
        self.agent_selector = lambda: random.randrange(num_agent)
        self.curr_agent = self.agent_selector()

    def observation_space(self, agent):
        return self.env.observation_space

    def action_space(self, agent):
        return self.env.action_space


class SingleToConcurrentMultiAgentWrapper(gym.core.Wrapper):
    """Single to Concurrent MultiAgent Wrapper"""

    agents: List[str]

    # TODO: Optionally provide the initial states
    def __init__(self, env, num_agent: int):
        super().__init__(env)
        self.agents = [f"agent{i}" for i in range(num_agent)]

    def observation_space(self, agent):
        return self.env.observation_space

    def action_space(self, agent):
        return self.env.action_space

    def reset(self):
        observations = {agent: None for agent in self.agents}
        return observations

    def step(self, actions: Dict):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            return {}

        # current observation is just the other player's most recent action
        observations = {a: actions[a] for a in self.agents}

        return observations

    def render(self):
        pass


register(id="MiniGrid-TSP-v0", entry_point="tpossp.minigrid:TSPEnv")

register(
    id="MiniGrid-TSPBenchmarkEnv-v0", entry_point="tpossp.minigrid:TSPBenchmarkEnv"
)


register(
    id="MiniGrid-TSPMultiFourGrids-v0", entry_point="tpossp.minigrid:TSPMultiFourGrids"
)


register(
    id="MiniGrid-AircraftTurnaroundEnv-v0",
    entry_point="tpossp.minigrid:AircraftTurnaroundEnv",
)


register(
    id="MiniGrid-AspenAircraftTurnaroundEnv-v0",
    entry_point="tpossp.minigrid:AspenAircraftTurnaroundEnv",
)
