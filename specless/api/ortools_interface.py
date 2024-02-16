"""
Usage
------

>> model = TSPModel()
>> # Apply changes to the model
>> solver = ORToolsSolver()
>> # Apply Changes to the solver
>> solution = solver.solve(model)


Supported Features (API)
----------------

Multiple Robots

>> model = TSPModel()
>> num_robot = 4
>> model.setNumRobot(num_robot)

Asymmetric Distances

E.g., one-way, varying speed

Timing Constraints

e.g., global, local time windows

Nonlinear costs

E.g., battery

Capacity

E.g. Carrying objects (vehicles, eggs), passengers

Repetitions

E.g., same tasks, multiple objects

Occupancy

E.g., sharing spaces, non-shareble space (1 robot at 1 location)

Probabilistic Events

E.g., Failures, MDP

Non-stochastic Events

E.g., road blockage, railway blockage

"""
from typing import List, Optional


class OrtoolsInterface:
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
