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
        -> Use OR-Tools
        -> https://developers.google.com/optimization/routing/penalties
    9. Pick and Delivery constraints?
        -> Use OR-Tools
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

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def distance(x1, y1, x2, y2):
    # Manhattan distance
    dist = abs(x1 - x2) + abs(y1 - y2)
    return dist


def get_distance_matrix(locations):
    num_locations = len(locations)

    matrix = {}
    for from_node in range(num_locations):
        matrix[from_node] = {}
        for to_node in range(num_locations):
            if from_node == to_node:
                matrix[from_node][to_node] = 0
            else:
                x1 = locations[from_node][0]
                y1 = locations[from_node][1]
                x2 = locations[to_node][0]
                y2 = locations[to_node][1]
                matrix[from_node][to_node] = distance(x1, y1, x2, y2)
    return matrix


def create_data_model():

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
        "Hallway 1": (3, 8),
        "Hallway 2": (5, 2),
        "Hallway 3": (5, 15),
        "Hallway 4": (7, 8),
    }

    #####################
    #       Define      #
    #####################
    rooms_to_visit: list[str] = [
        "Room B",
        "Room C",
        "Room J",
        "Room I",
        "Hallway 1",
        "Hallway 1",
        "Hallway 2",
        "Hallway 2",
        "Hallway 3",
        "Hallway 3",
        "Hallway 4",
        "Hallway 4",
    ]
    locations = [floormap[r] for r in rooms_to_visit]
    data = {}
    data["num_locations"] = len(locations)
    data["distance_matrix"] = get_distance_matrix(locations)
    data["demands"] = [5, 10, 4, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0]
    data["vehicle_capacities"] = [20, 10]
    data["num_vehicles"] = len(data["vehicle_capacities"])
    data["depot"] = 0
    data["optional_nodes"] = [1]
    # data["optional_nodes"] = [0]

    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    print(f"Objective: {assignment.ObjectiveValue()}")
    # Display dropped nodes.
    dropped_nodes = "Dropped nodes:"
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += f" {manager.IndexToNode(node)}"
    print(dropped_nodes)
    # Display routes
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total Distance of all routes: {total_distance}m")
    print(f"Total Load of all routes: {total_load}")


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        data["num_locations"], data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )
    # Allow to drop nodes.
    penalty = 1000
    # for node in range(1, data["num_locations"]):
    for node in data["optional_nodes"]:
        indices = [manager.NodeToIndex(node)]
        routing.AddDisjunction(indices, penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # )
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # )
    # search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)


if __name__ == "__main__":
    main()
