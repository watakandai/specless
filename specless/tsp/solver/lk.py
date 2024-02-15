from typing import List, Optional, Tuple

from specless.tsp.tsp import Node

from .base import TSPSolver, TSPWithTPOSolver


class LinKernighanTSPSolver(TSPSolver):
    def __init__(self):
        super().__init__()

    def solve(
        self,
        tsp,
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = True,
        timeout: int = 1,
    ) -> Tuple[List, float]:
        """Solve the VRP with time windows."""
        if init_nodes is None:
            init_nodes = [tsp.nodes[0]] * num_agent
        else:
            num_agent = len(init_nodes)

        raise NotImplementedError()
        # return [], 0.0


class LinKernighanTSPWithTPOSolver(TSPWithTPOSolver):
    def __init__(
        self,
    ):
        super().__init__()

    def solve(
        self,
        tsp,
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = True,
    ) -> Tuple[List, float]:
        """Solve the VRP with time windows."""
        if init_nodes is None:
            init_nodes = [tsp.nodes[0]] * num_agent
        else:
            num_agent = len(init_nodes)

        raise NotImplementedError()
        # return [], 0.0
