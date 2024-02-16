import random
from typing import List

import numpy as np

from specless.specification.base import Specification
from specless.specification.timed_partial_order import (
    TimedPartialOrder,
    generate_random_constraints,
)
from specless.tsp.solver.ortools import ORTSPSolver, ORTSPWithTPOSolver
from specless.tsp.tsp import TSP, TSPWithTPO


def test_ortools_on_tsp():
    # Number of locations
    nodes: list[int] = [0, 1, 2, 3]
    # Travel time
    costs: List[List[float]] = [
        [0, 3, 4, 5],
        [3, 0, 5, 4],
        [4, 5, 0, 3],
        [5, 4, 3, 0],
    ]
    # 1. Just Test with the cost (TSP Solver)
    tsp = TSP(nodes, costs)
    solver = ORTSPSolver()
    tour, cost = solver.solve(tsp)
    print("-" * 100)
    print(tour, cost)
    print("-" * 100)


def test_ortools_on_tsp_with_tpo():
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
    solver = ORTSPWithTPOSolver()
    tours, cost = solver.solve(tsp_with_tpo)
    assert tours[0][0] == 0
    assert tours[0][1] == 1
    assert tours[0][2] == 2
    assert tours[0][3] == 3
    assert tours[0][4] == 0
    assert cost == 18.0


def get_random_locations(N, width, height):
    coordinates = set()
    while len(coordinates) < N:
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
        coordinate = (x, y)
        coordinates.add(coordinate)
    return list(coordinates)


def test_ortools_on_random_tsp_with_tpo():
    width = 100
    height = 100
    num_nodes: int = 20

    nodes = list(range(num_nodes))
    ls = get_random_locations(num_nodes, width, height)
    ls = list(map(np.array, ls))
    costs = [[int(np.linalg.norm(l2 - l1)) for l2 in ls] for l1 in ls]

    initial_nodes = [nodes[0]]
    num_constraint_ratio: float = 0.2
    random_time_gap = lambda: random.randint(1, 10)
    tpo = generate_random_constraints(
        nodes,
        initial_nodes,
        costs,
        num_constraint_ratio,
        time_gap_callback=random_time_gap,
    )

    tsp_with_tpo = TSPWithTPO(nodes, costs, tpo)

    solver = ORTSPWithTPOSolver()
    tours, cost = solver.solve(tsp_with_tpo)
    assert len(tours) > 0
