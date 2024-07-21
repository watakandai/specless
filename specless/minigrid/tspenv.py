import random
from typing import Tuple

import distinctipy
import gymnasium as gym
from gym_minigrid.minigrid import Floor, Grid, MiniGridEnv, MissionSpace

# from .core import NoDirectionAgentGrid

MINIGRID_TO_GRAPHVIZ_COLOR = {
    "red": "firebrick",
    "green": "darkseagreen1",
    "blue": "steelblue1",
    "purple": "mediumpurple1",
    "yellow": "yellow",
    "grey": "gray60",
}


class TSPEnv(MiniGridEnv):
    """TSP Environment with Multiple floor locations with duplicate colors"""

    def __init__(
        self,
        num_locations: int = 5,
        width: int = 6,
        height: int = 6,
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        seed=None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.directionless_agent = True

        random.seed(seed)
        self.locations = self.get_random_locations(num_locations, width, height)

        super().__init__(
            mission_space=MissionSpace(lambda: "Visit each location once"),
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs,
        )

    def get_random_locations(self, N, width, height):
        coordinates = set([self.agent_start_pos])
        while len(coordinates) < N + 1:
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            coordinate: tuple[int, int] = (x, y)
            coordinates.add(coordinate)
        coordinates.remove(self.agent_start_pos)
        return list(coordinates)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for i, loc in enumerate(self.locations):
            ind = i % len(MINIGRID_TO_GRAPHVIZ_COLOR)
            color = list(MINIGRID_TO_GRAPHVIZ_COLOR.keys())[ind]
            self.put_obj(Floor(color=color), *loc)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Get to a green and a purple tiles"


class TSPBenchmarkEnv(MiniGridEnv):
    """TSP Environment with Multiple floor locations with duplicate colors"""

    def __init__(
        self,
        num_locations=5,
        width=20,
        height=20,
        agent_start_pos=(1, 5),
        agent_start_dir=0,
        seed=None,
        **kwargs,
    ):
        rgbs = distinctipy.get_colors(num_locations)
        f = lambda v: int(255 * v)
        rgbs = [tuple(map(f, rgb)) for rgb in rgbs]
        self.rgbs = rgbs
        random.seed(seed)
        self.locations = self.get_random_locations(num_locations, width, height)
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.directionless_agent = True

        mission_space = MissionSpace(mission_func=lambda: "Visit all red floors")
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=4 * width * height,
            see_through_walls=True,
            **kwargs,
        )

    def set_agent_start_pos(self, agent_start_pos):
        self.agent_start_pos = agent_start_pos

    def set_agent_start_dir(self, agent_start_dir):
        self.agent_start_dir = agent_start_dir

    def get_random_locations(self, N, width, height):
        coordinates = set()
        while len(coordinates) < N:
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            coordinate = (x, y)
            coordinates.add(coordinate)
        return list(coordinates)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # f = lambda v: int(255 * v)
        for i, loc in enumerate(self.locations):
            # rgb = self.rgbs[i]
            self.put_obj(Floor("red"), *loc)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Get to a green and a purple tiles"


gym.register(id="MiniGrid-TSP-v0", entry_point="specless.minigrid.tspenv:TSPEnv")


gym.register(
    id="MiniGrid-TSPBenchmarkEnv-v0",
    entry_point="specless.minigrid.tspenv:TSPBenchmarkEnv",
)
