import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import random
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from ortools.constraint_solver import routing_enums_pb2

from specless.specification.timed_partial_order import (
    generate_random_constraints,
)
from specless.tsp.solver.milp import MILPTSPWithTPOSolver
from specless.tsp.solver.ortools import ORTSPWithTPOSolver
from specless.tsp.tsp import TSPWithTPO
from specless.utils.benchmark import BenchmarkLogger

ARGS_TO_TPO: Dict[Tuple, TSPWithTPO] = {}


def get_random_locations(N, width, height) -> List[Tuple[int, int]]:
    coordinates: set = set()
    while len(coordinates) < N:
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
        coordinate = (x, y)
        coordinates.add(coordinate)
    return list(coordinates)


def test_ortools_on_random_tsp_with_tpo(
    num_nodes: int = 20,
    num_constraint_ratio: float = 0.2,
    max_time_gap: int = 10,
    num_agent: int = 1,
    solver_name: str = "milp",
    ith_run: int = 1,
    width: int = 100,
    height: int = 100,
) -> Tuple:
    args_key: Tuple[int, int, float, int] = (
        ith_run,
        num_nodes,
        num_constraint_ratio,
        max_time_gap,
    )

    # Note: This lets all the solvers to use the exact same TSP w/ TPO model.
    if args_key in ARGS_TO_TPO:
        tsp_with_tpo = ARGS_TO_TPO[args_key]
    else:
        nodes = list(range(0, num_nodes))

        locations: List[Tuple[int, int]] = get_random_locations(
            num_nodes, width, height
        )
        ls: List[np.ndarray] = list(map(np.array, locations))
        dist_matrix = [[int(np.linalg.norm(l2 - l1)) for l2 in ls] for l1 in ls]

        initial_nodes: List[int] = [nodes[0]]

        def random_time_gap():
            return random.randint(1, max_time_gap)

        tpo = generate_random_constraints(
            nodes,
            initial_nodes,
            dist_matrix,
            num_constraint_ratio,
            time_gap_callback=random_time_gap,
        )
        tsp_with_tpo = TSPWithTPO(nodes, dist_matrix, tpo)

        ARGS_TO_TPO[args_key] = tsp_with_tpo

    start = time.time()
    if solver_name == "milp":
        solver = MILPTSPWithTPOSolver()
    else:
        solver = ORTSPWithTPOSolver(
            first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
            timeout=60 * 10,
            solution_limit=1,
        )
    tours, cost = solver.solve(tsp_with_tpo, num_agent=num_agent)
    end = time.time()

    return tours, cost, end - start


if __name__ == "__main__":
    experiment_func: Callable[[Any], Tuple] = test_ortools_on_random_tsp_with_tpo
    arg_dict: Dict[str, List] = {
        "Node": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "ConstraintRatio": [0.5],
        "MaxTimeGap": [10, 40],
        "Agent": [1, 4],
        "Solver": ["milp"],
        "Iteration": [1],
    }
    return_key_strs: List[str] = ["Tours", "Cost", "Time[s]"]
    csvfilepath: str = str(Path(__file__).with_suffix("").with_suffix(".csv"))

    logger = BenchmarkLogger()
    logger.start(experiment_func, arg_dict, return_key_strs, csvfilepath)
