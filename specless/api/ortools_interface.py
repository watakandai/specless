"""
=================
Ortools Interface
=================

The Environment Setting Must be Structured and Programmable.
For now, the properties are mainly used during solving tsp. Not for RL purposes.
# TODO:
Node Capacity,
Edge Capacity,
Node Occupancy,
Edge Occupancy,

# Env Global Properties

>>> envbuilder = EnvBuilder() # Should have default params
>>> envbuilder.set_height(30)
>>> envbuilder.set_width(30)
>>> envbuilder.set_actions(num_direction=8, directionless=true)

# Objects

>>> envbuilder.add_object(10, 2, Floor("red"), "redfloor", maxRobot=3)

# or

>>> envbuilder.add_goal(10, 2, "A", maxRobot=1)
>>> envbuilder.add_goal(2, 25, "B", maxRobot=1)
>>> envbuilder.add_goal(25, 5, "C", maxRobot=1)
>>> envbuilder.add_goal(5, 25, "D", maxRobot=1)
>>> envbuilder.add_goal(2, 2, "E", maxRobot=1)
>>> envbuilder.add_goal(5, 2, "F", maxRobot=1)
>>> envbuilder.add_goal(1, 1, "G", maxRobot=1)

# Agents

>>> envbuilder.add_agent("Chapin", start=(1, 1), end=None,
...                     velocity=1, service_velocity=1,
...                     init_payload=0, max_payload=15,
...                     init_battery=100, min_battery=10,
...                     battery_speed=-2)

>>> envbuilder.add_agent("Mozart", start=None, end=(20, 2),
...                     velocity=5, service_velocity=1,
...                     init_payload=0, max_payload=10,
...                     init_battery=100, min_battery=10,
...                     battery_speed=-4)

>>> envbuilder.add_agent("Bach", start=(5, 25), end=(5, 25),
...                     velocity=10, service_velocity=1,
...                     init_payload=0, max_payload=5,
...                     init_battery=100, min_battery=10,
...                     battery_speed=-8)

# If start is uncertain, you can also provide a list.

>>> envbuilder.add_agent("", start=[(1, 1), (5, 10), (10, 5)], ...)
>>> envbuilder.add_agent("", end=[(1, 1), (5, 10), (10, 5)], ...)
>>> env = envbuilder.build()

>>> specbuilder = SpecificationBuilder()

# Tasks

>>> specbuilder.add_task("Visit A", location="A", assignments=["Bach"], payload=1)
>>> specbuilder.add_task("Task at B", location="B", standalone=False, service_cost=10)
>>> specbuilder.add_pickAndDelivery("Deliver from C to D", pickup="C", delivery="D")
>>> specbuilder.add_task("Visit E", location="E", payload=8)
>>> specbuilder.add_task("Visit D", location="D", payload=2)
>>> specbuilder.add_task("Visit F", location="F", payload=4)
>>> specbuilder.add_task("Machinery G", start="G", end="G", standalone=True, service_time=8)
>>> specbuilder.add_pickAndDelivery("Deliver from G to A", pickup="G", delivery="A")
>>> specbuilder.add_repeats(["Visit A", "Task at B"], 4)

# Order & Timing Constraints (You can impose time between tasks OR locations)

>>> specbuilder.add_local_time_constraint("Task A", "Task B", 0, 20)
>>> specbuilder.add_local_time_constraint("C", "D", 0, 30)
>>> specbuilder.add_global_time_constraint("Task F", 15, 100)
>>> specbuilder.add_precedent_constraint("Task B", "D")
>>> specbuilder.add_precedent_constraint("Machinery G", "Deliver from G to A")

# Capacity Constraint (These must be called)

>>> specbuilder.add_capacity_constraint("payload", "max")
>>> specbuilder.add_capacity_constraint("battery", "min")
>>> specbuilder.add_occupancy_constraint("numRobot", "max")
>>> specification = specbuilder.build()


>>> strategy = sl.TSPSynthesisAlgorithm().synthesize(env, specification)

>>> states, actions, video_path = sl.simulate(
...      env,
...      straetgy,
...      record_video=True,
...      video_folder=os.path.join(os.getcwd(), "videos"),
... )
"""

from typing import List, Optional


class OrtoolsInterface:
    """Ortools Interface for Robotics Task Allocation and Scheduling

    This class is a wrapper around the ortools library for solving robotics task allocation and scheduling problems.

    Parameters
    ----------
    weights : List[List[int]]
        A list of lists of integers representing the weights of the edges in the graph.
    num_robot : int, optional
        The number of robots in the system. Defaults to 1.
    starts : List[int], optional
        A list of integers representing the starting nodes for each robot. Defaults to None.
    ends : List[int], optional
        A list of integers representing the ending nodes for each robot. Defaults to None.

    Examples
    --------
    >>> weights = [[1, 2, 3], [4, 5, 6]]
    >>> model = OrtoolsInterface(weights)

    Notes
    -----
    The weights must be a list of lists of integers.
    """

    def __init__(
        self,
        weights: List[List[int]],
        num_robot: int = 1,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
    ) -> None:
        """
        Default to 1 robot and no starts or ends.
        Only int weights are supported

        Args:
            weights (List[List[int]]): _description_
            num_robot (int, optional): _description_. Defaults to 1.
            start (int, optional): _description_. Defaults to None.
            end (int, optional): _description_. Defaults to None.
        """
        if starts is None:
            # Create a dummy node with 0 edge weights to all nodes
            pass
        else:
            assert len(starts) == num_robot

        if ends is None:
            # Create a dummy node with 0 edge weights to all nodes
            pass
        else:
            assert len(ends) == num_robot

        raise NotImplementedError()

    # TODO: Think if this is necessary:
    def __addEdge(self, src: int, tgt: int, weight: int):
        raise NotImplementedError()

    def set_node_weight(self, n: int, weight: float):
        raise NotImplementedError()

    def set_edge_capacity(self, src, tgt, capacity: int, name: str):
        raise NotImplementedError()

    def set_node_capacity(self, n, capacity, name):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()


class RoboticsTaskModel:
    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def from_jsonfile(cls, filepath: str):
        # Load the json file

        # parse it
        # kwargs = {}
        # return cls(**kwargs)
        raise NotImplementedError()

    def add_location(self, location: str):
        raise NotImplementedError()

    def add_travel_time(self, src_location: str, tgt_location: str, travel_time: int):
        raise NotImplementedError()

    def add_service_time(self, location: str, service_time: int):
        raise NotImplementedError()

    def add_blockage(self, src_location: str, tgt_location: str):
        raise NotImplementedError()

    # TODO: Come up with what types of tasks there are
    def add_task(self):
        raise NotImplementedError()

    # TODO:
    def add_stationary_task(self):
        raise NotImplementedError()

    def add_capacity(self):
        pass

    def add_looping_task(self, task, num_loops: int):
        pass
