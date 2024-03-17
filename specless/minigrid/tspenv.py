import random

import distinctipy
import gymnasium as gym
from gym_minigrid.minigrid import Floor, Grid, Lava, MiniGridEnv, MissionSpace

# from .core import NoDirectionAgentGrid

MINIGRID_TO_GRAPHVIZ_COLOR = {}


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
        # if self.directionless_agent:
        #     self.grid = NoDirectionAgentGrid(width, height)
        # else:
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
        self.rgbs = rgbs
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
        # if self.directionless_agent:
        #     self.grid = NoDirectionAgentGrid(width, height)
        # else:
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


gym.register(id="MiniGrid-TSP-v0", entry_point="specless.minigrid.tspenv:TSPEnv")


gym.register(
    id="MiniGrid-TSPBenchmarkEnv-v0",
    entry_point="specless.minigrid.tspenv:TSPBenchmarkEnv",
)
