import unittest
from typing import List

from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.tsp.tsp import TSP, TSPWithTPO


def test_tsp():
    # Number of locations
    nodes: List[int] = [0, 1, 2, 3]
    # Travel time
    costs: List[List[float]] = [
        [0, 3, 4, 5],
        [3, 0, 5, 4],
        [4, 5, 0, 3],
        [5, 4, 3, 0],
    ]
    # 1. Just Test with the cost (TSP Solver)
    tsp = TSP(nodes, costs)
    assert isinstance(tsp, TSP)
    unittest.TestCase.assertListEqual(tsp.nodes, nodes)
    unittest.TestCase.assertListEqual(tsp.costs, costs)
    unittest.TestCase.assertListEqual(tsp.services, [0, 0, 0, 0])


def test_tsp_with_tpo():
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
    lb: float = 10
    ub: float = 20
    specification.add_local_constraint(src_node, tgt_node, lb, ub)

    tsp = TSPWithTPO(nodes, costs, specification)
    assert isinstance(tsp, TSPWithTPO)
