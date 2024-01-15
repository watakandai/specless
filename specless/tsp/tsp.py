"""
TSP
=====

# Number of locations
>> n = 4
>> nodes: List[int] = [0, 1, 2, 3]
# Ready time
>> a: List[float] = [0, 5, 0, 8]
# Due time
>> b: List[float] = [100, 16, 10, 14]
# Travel time
>> costs: List[List[float]] = [
    [0, 3, 4, 5],
    [3, 0, 5, 4],
    [4, 5, 0, 3],
    [5, 4, 3, 0],
]
# 1. Just Test with the cost (TSP Solver)
>> tsp = TSP(nodes, costs)

>> service_times = [2, 2, 2, 2]
>> tsp = TSP(nodes, costs, service_times)
"""
import itertools
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from specless.specification.timed_partial_order import TimedPartialOrder

Node = Any


class GTSP:
    """General TSP. Goal is to visit exactly 1 city from every set.
    Let V be a set of nodes (cities) and E as edges.
    Every node belongs to a set V_i \in [V_1, V_2, ..., V_n].

    Objective is
    """

    nodes: List[Node]
    edges: List[Tuple[Node, Node]]
    costs: List[List[float]]
    services: List[float]
    come_back_home: bool
    nodesets: List[List[Node]]
    node_set_mapping: Dict

    def __init__(
        self,
        nodes: List[Node],
        costs: List[List[float]],
        services: Optional[List[float]] = None,
        nodesets: Optional[List[List[Node]]] = None,
    ):
        if services is None:
            services = [0] * len(nodes)
        if nodesets is None:
            nodesets = [[n] for n in nodes]
        N: int = len(nodes)
        self.nodes = nodes
        self.edges = list(itertools.permutations(self.nodes, 2))
        self.costs = costs
        self.services = services
        self.nodesets = nodesets
        self.node_set_mapping = {
            n: i for i, nodeset in enumerate(nodesets) for n in nodeset
        }


class TSP(GTSP):
    """Traveling Salesman Problem"""

    def __init__(
        self,
        nodes: List[Node],
        costs: List[List[float]],
        services: Optional[List[float]] = None,
    ):
        super().__init__(nodes, costs, services)


class TSPTW(TSP):
    """Traveling Salesman Problem With Timed Window Constraints"""

    def __init__(
        self,
        nodes: List[Node],
        costs: List[List[float]],
        time_windows: Dict[int, Tuple[int, int]],
        services: Optional[List[float]] = None,
    ):
        super().__init__(nodes, costs, services)
        self.num_node = len(nodes)
        self.time_windows = time_windows

    @staticmethod
    def loadFile(file_path) -> Tuple:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Parse number of nodes
        num_nodes = int(lines[0])

        # Parse distance matrix
        distance_matrix = []
        for line in lines[1 : num_nodes + 1]:
            distances = list(map(int, line.split()))
            distance_matrix.append(distances)

        # Parse time windows
        time_windows = {}
        i = 0
        for line in lines[num_nodes + 1 :]:
            if line.strip() == "":
                break
            parts = line.split()
            time_window_start = int(parts[0])
            time_window_end = int(parts[1])
            time_windows[i] = (time_window_start, time_window_end)
            i += 1

        return num_nodes, distance_matrix, time_windows

    @classmethod
    def loadFromFile(cls, file_path) -> "TSPTW":
        return cls(*TSPTW.loadFile(file_path))

    def __str__(self):
        ret_str = ""
        ret_str += f"1 Depot + Number of nodes: {self.num_node-1}\n"
        ret_str += "Distance matrix: \n"
        for row in self.costs:
            ret_str += f"{row}\n"
        ret_str += "Time windows: \n"
        for node_id, (start, end) in self.time_windows.items():
            ret_str += f"Node {node_id}: ({start}, {end})\n"
        return ret_str[:-1]

    def satisfy(self, timed_trace: List[Tuple[str, float]], print_reason: bool = False):
        for e, t in timed_trace:
            lb, ub = self.time_windows[e]
            satisfy = lb <= t and t <= ub
            if not satisfy:
                if print_reason:
                    print(f"Node {e} at {t} did not satisfy {lb} <= t <= {ub}")
                return False
        return True


class TSPWithTPO(TSP):
    """Traveling Salesman Problem With TPO Constraints"""

    def __init__(
        self,
        nodes: List[Node],
        costs: List[List[float]],
        tpo: TimedPartialOrder,
        services: Optional[List[float]] = None,
    ):
        super().__init__(nodes, costs, services)
        self.num_node: int = len(nodes)
        self.tpo: TimedPartialOrder = tpo

    @classmethod
    def from_tsp(cls, tsp: TSP, tpo: TimedPartialOrder) -> Type["TSPWithTPO"]:
        tsp_with_tpo = cls(tsp.nodes, tsp.costs, tpo, services=tsp.services)
        tsp_with_tpo.nodesets = tsp.nodesets
        return tsp_with_tpo

    def satisfy(self, timed_trace, print_reason: bool = False):
        return self.tpo.satisfy(timed_trace, print_reason)


if __name__ == "__main__":
    # Number of locations
    n = 4
    nodes: List[int] = [0, 1, 2, 3]
    # Ready time
    a: List[float] = [0, 5, 0, 8]
    # Due time
    b: List[float] = [100, 16, 10, 14]
    # Travel time
    costs: List[List[float]] = [
        [0, 3, 4, 5],
        [3, 0, 5, 4],
        [4, 5, 0, 3],
        [5, 4, 3, 0],
    ]
    # 1. Just Test with the cost (TSP Solver)
    tsp = TSP(nodes, costs)

    benchmark = "Dumas"
    instance_name = "n20w20.001.txt"
    objective = "makespan"  # "traveltime"

    # Option 1: Load them Separately
    workspace = Path(__file__).parent.parent
    benchmark_path = workspace.joinpath("benchmark")
    file_path = benchmark_path.joinpath(f"{benchmark}/{instance_name}")
    solution_file_path = benchmark_path.joinpath(
        f"{benchmark}-best-known-{objective}.txt"
    )
    tsptw = TSPTW.loadFromFile(file_path)
    print(tsptw)
