from typing import List, Optional, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp

from specless.tsp.tsp import TSP, Node

from .base import TSPSolver, TSPWithTPOSolver

"""Extend it to Multi-Agents & Set"""

SOLUTION_STATUS = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
    3: "ROUTING_FAIL",
    4: "ROUTING_FAIL_TIMEOUT",
    5: "ROUTING_INVALID",
    6: "ROUTING_INFEASIBLE",
}


class ORTSPSolver(TSPSolver):
    def __init__(
        self,
        first_solution_strategy: Optional[int] = None,
        metaheuristic: Optional[int] = None,
        timeout: Optional[int] = None,
        solution_limit: Optional[int] = None,
    ):
        super().__init__()
        self.first_solution_strategy = first_solution_strategy
        self.metaheuristic = metaheuristic
        self.timeout = timeout
        self.solution_limit = solution_limit

    def create_data_model(self, tsp, num_agent, init_nodes):
        """Stores the data for the problem."""
        data = {}
        data["time_matrix"] = tsp.costs
        data["num_vehicles"] = num_agent
        data["depot"] = init_nodes
        return data

    def solve(
        self,
        tsp,
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = True,
    ) -> Tuple[List, float]:
        """Solve the VRP with time windows."""
        if init_nodes is None:
            init_nodes = [tsp.nodes[0]] * num_agent
        else:
            num_agent = len(init_nodes)
        # Instantiate the data problem.
        data = self.create_data_model(tsp, num_agent, init_nodes)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["time_matrix"]), data["num_vehicles"], data["depot"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["time_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        if self.first_solution_strategy is not None:
            search_parameters.first_solution_strategy = self.first_solution_strategy
        if self.metaheuristic is not None:
            search_parameters.local_search_metaheuristic = self.metaheuristic

        if self.timeout is not None:
            search_parameters.time_limit.seconds = self.timeout
        if self.solution_limit is not None:
            search_parameters.solution_limit = self.solution_limit

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        tours, cost = self.get_tours_and_cost(data, manager, routing, solution)
        return tours, cost

    def get_tours_and_cost(
        self, data, manager, routing, solution
    ) -> Tuple[List, float]:
        """Get tours and cost fromthe ortools solution"""
        tours: List = []
        cost: float = solution.ObjectiveValue()
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            tour = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                tour.append(node)
                index = solution.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            tour.append(node)
            tours.append(tour)
        return tours, cost


class ORTSPWithTPOSolver(TSPWithTPOSolver):
    def __init__(
        self,
        first_solution_strategy: Optional[int] = None,
        metaheuristic: Optional[int] = None,
        timeout: Optional[int] = None,
        solution_limit: Optional[int] = None,
    ):
        super().__init__()
        self.first_solution_strategy = first_solution_strategy
        self.metaheuristic = metaheuristic
        self.timeout = timeout
        self.solution_limit = solution_limit

    def create_data_model(self, tsp, num_agent, init_nodes):
        """Stores the data for the problem."""
        big_number = int(np.max(tsp.costs) * len(tsp.costs))
        data = {}
        data["time_matrix"] = tsp.costs
        data["time_windows"] = [
            (tsp.tpo.global_constraints[n]["lb"], tsp.tpo.global_constraints[n]["ub"])
            if n in tsp.tpo.global_constraints
            else (0, big_number)
            for n in tsp.nodes
        ]
        # src: {tgt: (lb, ub)}
        data["local_time_windows"] = {
            n: {
                tgt: (d["lb"], d["ub"])
                for tgt, d in tsp.tpo.local_constraints[n].items()
            }
            for n in tsp.nodes
            if n in tsp.tpo.local_constraints
        }
        data["num_vehicles"] = num_agent
        data["depot"] = init_nodes
        return data

    def solve(
        self,
        tsp,
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = True,
    ) -> Tuple[List, float]:
        """Solve the VRP with time windows."""
        if init_nodes is None:
            init_nodes = [tsp.nodes[0]] * num_agent
        else:
            num_agent = len(init_nodes)

        # Instantiate the data problem.
        data = self.create_data_model(tsp, num_agent, init_nodes)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["time_matrix"]), data["num_vehicles"], data["depot"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["time_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Time Windows constraint.
        time = "Time"
        # MaxCost * #Nodes
        maximum_time_per_vehicle: int = int(np.max(tsp.costs) * len(tsp.costs))
        routing.AddDimension(
            transit_callback_index,
            50,  # allow waiting time           # Max(ub-lb for all (lb,ub))
            maximum_time_per_vehicle,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time,
        )
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx in data["depot"]:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Add local time window constraints between locations
        for src, d in data["local_time_windows"].items():
            for tgt, bound in d.items():
                src_index = manager.NodeToIndex(src)
                tgt_index = manager.NodeToIndex(tgt)
                lb, ub = bound
                routing.solver().Add(
                    time_dimension.CumulVar(tgt_index)
                    - time_dimension.CumulVar(src_index)
                    >= lb
                )
                routing.solver().Add(
                    time_dimension.CumulVar(tgt_index)
                    - time_dimension.CumulVar(src_index)
                    <= ub
                )

        # Add time window constraints for each vehicle start node.
        for depot_idx in data["depot"]:
            for vehicle_id in range(data["num_vehicles"]):
                index = routing.Start(vehicle_id)
                time_dimension.CumulVar(index).SetRange(
                    data["time_windows"][depot_idx][0],
                    data["time_windows"][depot_idx][1],
                )

        # Instantiate route start and end times to produce feasible times.
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i))
            )
        time_dimension.SetGlobalSpanCostCoefficient(5000)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        if self.first_solution_strategy is not None:
            search_parameters.first_solution_strategy = self.first_solution_strategy

        if self.metaheuristic is not None:
            search_parameters.local_search_metaheuristic = self.metaheuristic

        if self.timeout is not None:
            search_parameters.time_limit.seconds = self.timeout
        if self.solution_limit is not None:
            search_parameters.solution_limit = self.solution_limit

        # search_parameters.use_light_propagation = False
        # To log the search uncomment the following.
        search_parameters.log_search = True

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        status = routing.status()
        print("STATUS: ", status, SOLUTION_STATUS[status])
        # print("STATUS: ", s, pywrapcp.RoutingModel.Status(s))

        # print("First Solution Strategy: ", search_parameters.first_solution_strategy)
        # print("Local Search Heuristic: ", search_parameters.local_search_metaheuristic)
        # print(
        #     "Selected First Solution Strategy: ",
        #     routing.GetAutomaticFirstSolutionStrategy(),
        # )

        # TODO: If status == ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED:
        # Then rerun with solution limit of 100?

        if status not in [1, 2]:
            return [], float("inf")
        tours, cost = self.get_tours_and_cost(data, manager, routing, solution)
        return tours, cost

    def get_tours_and_cost(
        self, data, manager, routing, solution
    ) -> Tuple[List, float]:
        """Get tours and cost fromthe ortools solution"""
        tours: List = []
        time_dimension = routing.GetDimensionOrDie("Time")
        total_time = 0
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            tour = []
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                node = manager.IndexToNode(index)
                tour.append(node)
                index = solution.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            tour.append(node)
            tours.append(tour)
            time_var = time_dimension.CumulVar(index)
            total_time += solution.Min(time_var)
        return tours, total_time


if __name__ == "__main__":
    # Number of locations
    nodes = [0, 1, 2, 3]
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
