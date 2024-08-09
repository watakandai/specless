"""
Goal
====
In this script, we will show how to solve/implement the examples

Examples
========
    1. Given number of rooms, solve a TSP for a single agent.
    2. Multiple Agent
    3. Global Time Constraints (and with the TPO representation)
    4. Local Time Constraints (and with the TPO representation)
    4.5. How to export the MILP formulation to a file for debugging?
    5. What if we don't need to come back to its home depot?
    6. What if robots are in a new location (not in the designated area)
        -> Run A* to get the cost matrix
    7. What if a path between two locations are blocked?
        -> Set the cost to a big number (not infinity)
    8.1. Can we visit the same place N times?
        -> Yes, specless can easily deal with the problem
    8.2. Can we visit the same place arbitrary many times?
        -> There are multiple ways: (1) Reformulate MILP, (2) Use OR-Tools
        -> https://developers.google.com/optimization/routing/penalties

    #! NOTE: The following constraints are difficult to model in our MILP formulation,
    # because we formulate the TSP as a flow constraining problem whic makes it difficult
    # to add additional "dimension" constraints, e.g., pickup, resource, and capacity.

    9. Pick and Delivery constraints
        #! NOTE: Pick and Delivery constraints are difficult to model in MILP,
        because it constraining the same vehicle to pickup and deliver is difficult.
        -> Not Possible. Use OR-Tools
        -> https://developers.google.com/optimization/routing/pickup_delivery
    10. Resource constraints, etc. Battery
        -> Use OR-Tools
        -> https://developers.google.com/optimization/routing/cvrptw_resources
    11. Carriege Capacity constraints, etc, weight of the package.
        -> Use OR-Tools
        -> https://developers.google.com/optimization/routing/cvrp

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

from examples.demo.planning_utils import get_location_assignments


def main():
    #####################
    #       Given       #
    #####################
    # A floormap
    floormap = {
        "Room A": (0, 5),
        "Room B": (0, 15),
        "Room C": (3, 0),
        "Room D": (4, 5),
        "Room E": (4, 10),
        "Room F": (6, 5),
        "Room G": (6, 10),
        "Room H": (7, 0),
        "Room I": (10, 5),
        "Room J": (10, 15),
    }

    # Compute the distance cost between two locations
    # For now, let's just use the euclidean distance for demo.
    # (In a continuous environment, we can similarly use A* to compute the distance)
    #! NOTE: We need to use A* for modeling more accurate distance of the environment
    dist = lambda v1, v2: ((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2) ** 0.5
    costs = [[dist(v1, v2) for v2 in floormap.values()] for v1 in floormap.values()]

    #####################
    #       Define      #
    #####################
    # Define a task (rooms to visit)
    rooms_to_visit = ["Room B", "Room C", "Room J", "Room I"]
    # Define initial locations of the robot
    robot_initial_locations = ["Room A"]
    rooms_of_interest: list[str] = rooms_to_visit + robot_initial_locations

    #####################
    #        Main       #
    #####################
    # Recreate the cost matrix
    rooms = list(floormap.keys())
    costs = [
        [costs[rooms.index(r1)][rooms.index(r2)] for r2 in rooms_of_interest]
        for r1 in rooms_of_interest
    ]

    tours, cost, timestamps = get_location_assignments(
        rooms_of_interest,
        robot_initial_locations,
        costs,
    )

    print(tours)
    print(cost)
    print(timestamps)


if __name__ == "__main__":
    main()
