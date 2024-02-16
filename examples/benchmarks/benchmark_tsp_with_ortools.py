import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import random
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from specless.tsp.solver.ortools import ORTSPSolver
from specless.tsp.tsp import TSP
from specless.utils.benchmark import BenchmarkLogger

ARGS_TO_TSP: Dict[Tuple, TSP] = {}


def get_random_locations(N, width, height) -> List[Tuple[int, int]]:
    coordinates: set = set()
    while len(coordinates) < N:
        x = random.randint(1, width - 2)
        y = random.randint(1, height - 2)
        coordinate = (x, y)
        coordinates.add(coordinate)
    return list(coordinates)


def test_ortools_on_random_tsp(
    num_nodes: int = 20,
    num_agent: int = 1,
    ith_run: int = 1,
    width: int = 100,
    height: int = 100,
) -> Tuple:
    args_key: Tuple[int, int, float, int] = (
        ith_run,
        num_nodes,
    )

    # Note: This lets all the solvers to use the exact same TSP model.
    if args_key in ARGS_TO_TSP:
        tsp = ARGS_TO_TSP[args_key]
    else:
        nodes = list(range(0, num_nodes))

        locations: List[Tuple[int, int]] = get_random_locations(
            num_nodes, width, height
        )
        ls: List[np.ndarray] = list(map(np.array, locations))
        dist_matrix = [[int(np.linalg.norm(l2 - l1)) for l2 in ls] for l1 in ls]
        tsp = TSP(nodes, dist_matrix)
        ARGS_TO_TSP[args_key] = tsp

    start = time.time()
    solver = ORTSPSolver(timeout=180)
    tours, cost = solver.solve(tsp, num_agent=num_agent)
    end = time.time()

    return tours, cost, end - start


if __name__ == "__main__":
    experiment_func: Callable[[Any], Tuple] = test_ortools_on_random_tsp
    arg_dict: Dict[str, List] = {
        "Node": [100],
        "Agent": [1, 4],
    }
    return_key_strs: List[str] = ["Tours", "Cost", "Time[s]"]
    csvfilepath: str = str(Path(__file__).with_suffix("").with_suffix(".csv"))

    logger = BenchmarkLogger()
    logger.start(experiment_func, arg_dict, return_key_strs, csvfilepath)
