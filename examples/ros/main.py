from typing import Dict, List, Tuple

import specless as sl

Location = Tuple[float, float]
TimeBound = Tuple[float, float]


def get_node_assignments(
    nodes: List[int],
    costs: List[List[float]],
    initial_nodes: List[int],
    num_agent: int = 1,
    global_constraints: Dict[int, Tuple[float, float]] = {},
    local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = {},
):
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


def get_location_assignments(
    locations: List[Location],
    costs: List[List[float]],
    initial_locations: List[Location],
    num_agent: int = 1,
    global_constraints: Dict[Location, TimeBound] = {},
    local_constraints: Dict[Tuple[Location, Location], TimeBound] = {},
):
    # Convert locations to nodes
    n: int = len(locations)
    nodes: List[int] = list(range(n))
    node_to_loc = {i: l for i, l in enumerate(locations)}
    loc_to_node = {l: i for i, l in enumerate(locations)}
    initial_nodes = [loc_to_node[l] for l in locations]
    global_constraints_ = {loc_to_node[l]: tb for l, tb in global_constraints.items()}
    local_constraints_ = {
        (loc_to_node[s], loc_to_node[t]): tb for (s, t), tb in local_constraints.items()
    }

    # Compute node assignments
    tours, cost, timestamps = get_node_assignments(
        nodes, costs, initial_nodes, num_agent, global_constraints_, local_constraints_
    )
    # Convert node assignments to location assignments
    location_assignments = list(map(lambda n: node_to_loc[n], tours))
    return location_assignments, cost, timestamps


def main():
    # Load files

    # TODO: Define the number of agents
    num_agent: int = 1
    # TODO: Define a list of locations
    locations = []
    # TODO: Define a list of travel costs between locations
    costs = [[]]
    # TODO: Define a list of global time constraints map at locations in the form of (LB, UB)
    global_constraints: Dict[Location, TimeBound] = {
        # ex) (1, 2): [0, 10] means a robot must reach at (1, 2) between 0 to 10 seconds since the simulation has started
    }

    # TODO: Define a list of local time constraints between locations in the form of (LB, UB)
    local_constraints: Dict[Tuple[Location, Location], TimeBound] = {
        # ex) ((1, 2), (5, 4)): [10, 20] means from location (1, 2) to (5, 4) it must reach in between 10 to 20 seconds
    }
    get_location_assignments(
        locations, costs, num_agent, global_constraints, local_constraints
    )
