from typing import Dict, List, Optional, Tuple

import numpy as np

import specless as sl

Location = Tuple[float, float]
TimeBound = Tuple[float, float]


def get_node_assignments(
    nodes: List[int],
    costs: List[List[float]],
    initial_nodes: List[int],
    global_constraints: Dict[int, Tuple[float, float]] = {},
    local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = {},
):
    num_agent: int = len(initial_nodes)

    # Define Time Specification
    tpo: sl.TimedPartialOrder = sl.TimedPartialOrder.from_constraints(
        global_constraints, local_constraints
    )

    # Construct a TSP instance
    tsp_with_tpo = sl.TSPWithTPO(nodes, costs, tpo)
    # Instantiate a solver
    tspsolver = sl.MILPTSPWithTPOSolver()
    # Solve TSP -> Tours
    tours, cost, timestamps = tspsolver.solve(
        tsp_with_tpo, num_agent=num_agent, init_nodes=initial_nodes
    )

    return tours, cost, timestamps


def get_location_assignments(
    locations: List[Location],
    initial_nodes: List[int],
    costs: Optional[List[List[float]]] = None,
    global_constraints: Dict[int, TimeBound] = {},
    local_constraints: Dict[Tuple[int, int], TimeBound] = {},
):
    # Convert locations to nodes
    n: int = len(locations)
    nodes: List[int] = list(range(n))
    node_to_loc = {i: l for i, l in enumerate(locations)}

    if costs is None:
        costs = [
            [np.linalg.norm(np.array(l2) - np.array(l1)) for l1 in locations]
            for l2 in locations
        ]

    # Compute node assignments
    tours, cost, timestamps = get_node_assignments(
        nodes, costs, initial_nodes, global_constraints, local_constraints
    )
    # Convert node assignments to location assignments
    location_assignments = [list(map(lambda n: node_to_loc[n], tour)) for tour in tours]
    return location_assignments, cost, timestamps


def main():
    # Load files

    # TODO: Define a list of locations
    locations = [(4, 4), (2, 0), (0, 2), (3, 3)]
    initial_nodes = [0]

    # TODO: OPTIONAL: Define a list of travel costs between locations if EUCLIDEAN distance is not used
    # costs = [[]]

    # TODO: Define a list of global time constraints map at locations in the form of (LB, UB)
    global_constraints: Dict[Location, TimeBound] = {
        # ex) (1, 2): [0, 10] means a robot must reach at (1, 2) between 0 to 10 seconds since the simulation has started
        0: (0, 100),
        1: (5, 16),
        2: (0, 10),
        3: (8, 14),
    }

    # TODO: Define a list of local time constraints between locations in the form of (LB, UB)
    local_constraints: Dict[Tuple[Location, Location], TimeBound] = {
        # ex) ((1, 2), (5, 4)): [10, 20] means from location (1, 2) to (5, 4) it must reach in between 10 to 20 seconds
        (1, 2): [3, 7],
    }

    get_location_assignments(
        locations,
        initial_nodes,
        global_constraints=global_constraints,
        local_constraints=local_constraints,
    )


if __name__ == "__main__":
    main()
