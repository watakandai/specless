from typing import List

from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.tsp.solver.milp import MILPTSPWithTPOSolver
from specless.tsp.tsp import TSPWithTPO


def test_milpsolver():
    # Number of locations
    nodes: List[int] = [0, 1, 2, 3]
    # Travel time
    costs: List[List[float]] = [
        [0, 3, 4, 5],
        [3, 0, 5, 4],
        [4, 5, 0, 3],
        [5, 4, 3, 0],
    ]
    # Ready time
    lbs: List[float] = [0, 5, 0, 8]
    # Due time
    ubs: List[float] = [100, 16, 10, 14]
    # Construct a TPO
    specification: Specification = TimedPartialOrder()
    for node, (lb, ub) in enumerate(zip(lbs, ubs)):
        specification.add_global_constraint(node, lb, ub)

    src_node: int = 1
    tgt_node: int = 2
    lb: float = 3
    ub: float = 7
    specification.add_local_constraint(src_node, tgt_node, lb, ub)

    tsp_with_tpo = TSPWithTPO(nodes, costs, specification)

    # Solve TSP -> Tours
    tspsolver = MILPTSPWithTPOSolver()
    tours, cost, timestamps = tspsolver.solve(tsp_with_tpo)

    assert tours[0][0] == 0
    assert tours[0][1] == 1
    assert tours[0][2] == 2
    assert tours[0][3] == 3
    assert tours[0][4] == 0
    assert cost == 18.0
