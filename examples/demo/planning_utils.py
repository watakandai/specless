"""
Examples
========
    1. Given number of rooms, solve a TSP for a single agent.
    2. Multiple Agent
    3. Global Time Constraints
    4. Local Time Constraints
    5. How to add robot initial locations?
    6. What if we don't need to come back to its home depot?
    7. What if robots are not in the new locations?
    8. What if a path between two locations are blocked?
    9. Can we visit the same place multiple times?
    10. Pick and Delivery constraints? -> Not Possible. Use OR-Tools
    11. Resource constraints, etc. Battery
    12. Carriege Capacity constraints, etc, weight of the package.

Environment
===========
Consider the following environment:

########################################################
#    #                        #                        #
#    #             A          #                  B     #
#    ############     #########################     ####
# C                                                    #
#    #    #######     ############      #####          #
#    #    #        D       #         E      #          #
#    #    #                #                #          #
######    ###################################          #
#    #    #                #                #          #
#    #    #        F       #         G      #          #
#    #    #######     ############      #####          #
# H                                                    #
#    ############     #########################     ####
#    #             I          #                  J     #
#    #                        #                        #
########################################################
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import specless as sl

Location = str
TimeBound = Tuple[float, float]


def from_str_to_id_tpo(timed_partial_order, loc_to_node):
    global_constraints = {}
    local_constraints = {}
    for loc, (lb, ub) in timed_partial_order.global_constraints.items():
        node = loc_to_node[loc]
        global_constraints[node] = (lb, ub)

    for loc1, srd_dict in timed_partial_order.local_constraints.items():
        for loc2, (lb, ub) in srd_dict.items():
            node1, node2 = loc_to_node[loc1], loc_to_node[loc2]
            local_constraints[(node1, node2)] = (lb, ub)

    return sl.TimedPartialOrder.from_constraints(global_constraints, local_constraints)


def get_node_assignments(
    nodes: List[int],
    costs: List[List[float]],
    initial_nodes: List[int],
    global_constraints: Dict[int, Tuple[float, float]] = {},
    local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = {},
    timed_partial_order: Optional[sl.TimedPartialOrder] = None,
    come_back_home: bool = True,
    export_filename: Optional[str] = None,
):
    num_agent: int = len(initial_nodes)

    # Define Time Specification
    if not timed_partial_order:
        timed_partial_order: sl.TimedPartialOrder = (
            sl.TimedPartialOrder.from_constraints(global_constraints, local_constraints)
        )

    # Construct a TSP instance
    tsp_with_tpo = sl.TSPWithTPO(nodes, costs, timed_partial_order)
    # Instantiate a solver
    tspsolver = sl.MILPTSPWithTPOSolver()
    # Solve TSP -> Tours
    tours, cost, timestamps = tspsolver.solve(
        tsp_with_tpo,
        num_agent=num_agent,
        init_nodes=initial_nodes,
        come_back_home=come_back_home,
        export_filename=export_filename,
    )

    return tours, cost, timestamps


def get_location_assignments(
    locations: List[Location],
    initial_locations: List[Location],
    costs: Optional[List[List[float]]] = None,
    global_constraints: Dict[Location, TimeBound] = {},
    local_constraints: Dict[Tuple[Location, Location], TimeBound] = {},
    timed_partial_order: Optional[sl.TimedPartialOrder] = None,
    come_back_home: bool = True,
    export_filename: Optional[str] = None,
):
    # Convert locations to nodes
    n: int = len(locations)
    nodes: List[int] = list(range(n))
    node_to_loc = {i: l for i, l in enumerate(locations)}
    loc_to_node = {l: i for i, l in enumerate(locations)}
    initial_nodes = [loc_to_node[l] for l in initial_locations]

    if timed_partial_order:
        # Change the keys from strings to node IDs
        timed_partial_order = from_str_to_id_tpo(timed_partial_order, loc_to_node)
        node_global_constraints = {}
        node_to_node_local_constraints = {}
    else:
        node_global_constraints: Dict[int, TimeBound] = {
            loc_to_node[loc]: bound for loc, bound in global_constraints.items()
        }
        node_to_node_local_constraints: Dict[Tuple[int, int], TimeBound] = {
            (loc_to_node[l1], loc_to_node[l2]): bound
            for (l1, l2), bound in local_constraints.items()
        }

    if costs is None:
        costs = [
            [np.linalg.norm(np.array(l2) - np.array(l1)) for l1 in locations]
            for l2 in locations
        ]

    # Compute node assignments
    tours, cost, timestamps = get_node_assignments(
        nodes,
        costs,
        initial_nodes,
        node_global_constraints,
        node_to_node_local_constraints,
        timed_partial_order,
        come_back_home,
        export_filename,
    )
    # Convert node assignments to location assignments
    location_assignments = [list(map(lambda n: node_to_loc[n], tour)) for tour in tours]
    return location_assignments, cost, timestamps
