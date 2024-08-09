import specless as sl  # or load from specless.inference import TPOInference
from examples.demo.planning_utils import get_location_assignments


def main():

    ### Timed Partial Order Inference

    # Manually prepare a list of demonstrations
    demonstrations: list = [
        [[1, "Room B"], [16, "Room C"], [31, "Room J"], [46, "Room I"]],
        [[10, "Room J"], [20, "Room I"], [30, "Room B"], [40, "Room C"]],
    ]

    # Timed Partial Order Inference
    inference = sl.TPOInferenceAlgorithm()
    timed_partial_order: sl.Specification = inference.infer(demonstrations)

    print(timed_partial_order)

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
    costs = [[dist(v1, v2) for v2 in floormap.values()] for v1 in floormap.values()]

    #####################
    #       Define      #
    #####################
    # Define a task (rooms to visit)
    rooms_to_visit = ["Room B", "Room C", "Room J", "Room I"]
    # Define initial locations of the robot
    robot_initial_locations = ["Room A", "Room D"]
    rooms_of_interest = rooms_to_visit + robot_initial_locations

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
        timed_partial_order=timed_partial_order,
    )

    print(tours)
    print(cost)
    print(timestamps)


if __name__ == "__main__":
    main()
