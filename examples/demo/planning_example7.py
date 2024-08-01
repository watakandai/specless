"""
Goal
====
In this script, we will show how to solve/implement for Example 1,2,3,4

Examples
========
    1. Given number of rooms, solve a TSP for a single agent.
    2. Multiple Agent
    3. Global Time Constraints
    4. Local Time Constraints
    5. What if we don't need to come back to its home depot?
    6. What if robots are in a new location (not in the designated area)
        -> Run A* to get the cost matrix
    7. What if a path between two locations are blocked?
        -> Set the cost to a big number (not infinity)
    8. Can we visit the same place multiple times?'
        -> There are multiple ways: (1) Reformulate MILP, (2) Use OR-Tools
        -> https://developers.google.com/optimization/routing/penalties
    9. Pick and Delivery constraints?
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

from collections import defaultdict
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
    # Distance cost between two locations
    # We can similarly use A* to compute the "actual" distance
    dist = lambda v1, v2: ((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2) ** 0.5
    # For now, let's just use the euclidean distance
    cost_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for v1 in floormap.values():
        for v2 in floormap.values():
            # NOTE: Set a big number
            if v1 == (4, 5) and v2 == (4, 10):
                #! NOTE: For example
                cost_dict[v1][v2] = 1000
            else:
                cost_dict[v1][v2] = dist(v1, v2)

    #####################
    #       Define      #
    #####################
    # Define a task (rooms to visit)
    rooms_to_visit: list[str] = ["Room B", "Room C", "Room J", "Room I"]
    # Define initial locations of the robot
    robot_curr_coords = [(3, 15), (7, 15)]

    #####################
    #        Main       #
    #####################
    # Update the Floormap
    n = len(robot_curr_coords)
    robot_locations = [f"Robot{i}" for i in range(n)]
    for l, coord in zip(robot_locations, robot_curr_coords):
        floormap[l] = coord

    # Update the cost dict
    for l, coord in zip(robot_locations, robot_curr_coords):
        floormap[l] = coord
        for v in floormap.values():
            cost_dict[coord][v] = dist(coord, v)
            cost_dict[v][coord] = dist(v, coord)

    # Recreate the cost matrix
    rooms_of_interest = rooms_to_visit + robot_locations
    rooms = list(floormap.keys())
    costs = [
        [cost_dict[floormap[r1]][floormap[r2]] for r2 in rooms_of_interest]
        for r1 in rooms_of_interest
    ]

    tours, cost, timestamps = get_location_assignments(
        rooms_of_interest,
        robot_locations,
        costs,
        come_back_home=False,
    )

    print(tours)
    print(cost)
    print(timestamps)


if __name__ == "__main__":
    main()
