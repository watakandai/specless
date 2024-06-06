import gymnasium as gym
from gym_minigrid.minigrid import (
    COLORS,
    Box,
    Floor,
    Grid,
    Lava,
    MiniGridEnv,
    MissionSpace,
    fill_coords,
    point_in_rect,
)


class MyBox(Box):
    def __init__(
        self,
        color,
        contains=None,
    ):
        super().__init__(color, contains)

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
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            mission_space=MissionSpace(lambda: "Visit each location once"),
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs,
        )

    def _gen_grid(self, width, height):
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
        self.put_obj(MyBox("purple"), 5, 12)
        # Catering & Cleaning Station
        self.put_obj(Floor("purple"), 2, 19)

        # Conveyor
        self.put_obj(MyBox("grey"), 7, 12)
        # Conveyor
        self.put_obj(Floor("grey"), 11, 19)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Visit All Areas"


gym.register(
    id="MiniGrid-AircraftTurnaroundEnv-v0",
    entry_point="specless.minigrid.aircraftenv:AircraftTurnaroundEnv",
)
