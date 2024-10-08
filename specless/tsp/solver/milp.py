"""

>>> import specless as sl

>>> n = 4
>>> nodes: List[int] = [0, 1, 2, 3]

# Ready time

>>> a: List[float] = [0, 5, 0, 8]

# Due time

>>> b: List[float] = [100, 16, 10, 14]

# Travel time

>>> costs: List[List[float]] = [
...    [0, 3, 4, 5],
...    [3, 0, 5, 4],
...    [4, 5, 0, 3],
...    [5, 4, 3, 0],
... ]
>>> tsp = sl.TSP(nodes, costs)
>>> tspsolver = sl.MILPTSPSolver()
>>> tours, costs = tspsolver.solve(tsp) # doctest: +ELLIPSIS
Restricted license...
>>> tours
[[0, 2, 3, 1, 0]]
>>> costs # doctest: +ELLIPSIS
13.99...
"""

import copy
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from specless.tsp.tsp import TSP, Node

from .base import TSPSolver, TSPWithTPOSolver

workspace = Path(__file__).parent.parent.parent


"""Extend it to Multi-Agents & Set"""


class MILPTSPSolver(TSPSolver):
    def __init__(self):
        super().__init__()

    def initialize_problem(self, tsp):
        """Initialize the problem"""
        env = gp.Env()
        # env.setParam("TimeLimit", 30 * 60)
        env.setParam("OutputFlag", 0)

        # Declare and initialize model
        m = gp.Model(env=env)
        m.setParam("OutputFlag", False)
        # Create decision variables for choosing edges
        x = m.addVars(tsp.edges, vtype=GRB.BINARY, name="edges")
        # Create continuous time variables [1, n]
        t = m.addVars(tsp.nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="times")
        # Create continuous time variables for the TERMINAL nodes
        init_nodes = [(i, n) for i, n in enumerate(self.init_nodes)]
        tT = m.addVars(init_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="timesTerminal")
        tf = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tFinal")

        variables = {"x": x, "t": t, "tT": tT, "tf": tf}
        return m, variables

    def optimize(self, m, variables, objective: gp.LinExpr):
        """Optimize the defined model"""
        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()

    def get_tours(self, m, variables):
        x = variables["x"]
        vals = m.getAttr("X", x)
        edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tours = []
        for init_node in set(self.init_nodes):
            for i, j in edges.select(init_node, "*"):
                current = j
                tour = [init_node]
                while current not in self.init_nodes:
                    tour.append(current)
                    next = [j for i, j in edges.select(current, "*")][0]
                    current = next
                tour.append(init_node)
                tours.append(tour)

        if self.come_back_home:
            return tours
        return list(map(lambda x: x[0:-1], tours))

    def get_cost(self, m):
        return float(m.objVal)

    def solve(
        self,
        tsp: TSP,
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = True,
        export_filename: Optional[str] = None,
    ) -> Tuple[List, float]:
        tsp = copy.deepcopy(tsp)

        if init_nodes is None:
            init_nodes = [tsp.nodes[0]] * num_agent
        else:
            num_agent = len(init_nodes)

        self.agents = list(range(num_agent))
        self.init_nodes = init_nodes
        self.come_back_home: bool = come_back_home

        # If we are not coming back homes, we ignore the last traveling cost.
        if not self.come_back_home:
            for init_node in init_nodes:
                for i in set(tsp.nodes) - set([init_node]):
                    tsp.costs[i][init_node] = 0

        # Initialize the Problem
        # Users can add variables m.addVars() and append to variables.update({}) if wanted.
        m, variables = self.initialize_problem(tsp)

        # Users can add their own function to add more constraints
        x = variables["x"]
        t = variables["t"]
        tT = variables["tT"]
        tf = variables["tf"]
        non_init_nodes = list(set(tsp.nodes) - set(self.init_nodes))

        for nodes in tsp.nodesets:
            K = sum(init_node in nodes for init_node in self.init_nodes)
            # If the node set includes more initial nodes,
            # then set the #incoming/outgoing edge to the #init_nodes in that set (K)
            # If not exist, just set #in/outgoing edges to 1
            if K == 0:
                K = 1
            # Incoming and Outgoing FLow must be equal at these edges
            m.addConstr(gp.quicksum(x.sum("*", n) for n in nodes) == K, "incoming")
            m.addConstr(gp.quicksum(x.sum(n, "*") for n in nodes) == K, "outgoing")

        # The flow conservation must be enforced to aaaaalllll nodes.
        m.addConstrs((x.sum("*", n) == x.sum(n, "*") for n in tsp.nodes), "flow")

        # Time Delays by taking an edge i,j
        m.addConstrs(
            (
                (x[(i, j)] == 1) >> (t[j] - t[i] >= tsp.costs[i][j])
                for (i, j) in tsp.edges
                if j not in self.init_nodes
            ),
            "delay",
        )

        for ii, I in enumerate(self.init_nodes):
            m.addConstrs(
                (
                    (x[(i, I)] == 1) >> (tT[(ii, I)] - t[i] >= tsp.costs[i][I])
                    for i in non_init_nodes
                ),
                "delayTerm",
            )

        # Users can add a new objective by replacing get_edge_cost_objective
        m.addGenConstrMax(tf, tT)
        self.optimize(m, variables, tf)
        if export_filename:
            m.write(export_filename)

        if m.status != GRB.OPTIMAL:
            print("Tour: [], Cost: n/a")
            sys.exit()

        for var in m.getVars():
            if abs(var.x) > 1e-6:
                print("{0}: {1}".format(var.varName, var.x))
        print("Total matching score: {0}".format(m.objVal))

        tours = self.get_tours(m, variables)
        cost = self.get_cost(m)
        return tours, cost


class MILPTSPWithTPOSolver(TSPWithTPOSolver):
    def __init__(
        self,
    ):
        super().__init__()

    def initialize_problem(self, tsp):
        """Initialize the problem"""
        # Declare and initialize model
        env = gp.Env()
        env.setParam("TimeLimit", 30 * 60)
        env.setParam("OutputFlag", 0)

        m = gp.Model(env=env)
        # Create decision variables for choosing edges
        x = m.addVars(tsp.edges, vtype=GRB.BINARY, name="edges")
        # Create continuous time variables [1, n]
        t = m.addVars(tsp.nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="times")
        # Create continuous time variables for the TERMINAL nodes
        init_nodes: list[tuple[int, Node]] = [
            (i, n) for i, n in enumerate(self.init_nodes)
        ]
        tT = m.addVars(init_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="timesTerminal")
        tf = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tFinal")

        variables = {"x": x, "t": t, "tT": tT, "tf": tf}

        return m, variables

    def optimize(self, m, variables, objective: gp.LinExpr):
        """Optimize the defined model"""
        # print("Objective Type: ", type(objective))
        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()

    def get_tours(self, m, variables):
        x = variables["x"]
        vals = m.getAttr("X", x)
        edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tours = []
        for init_node in set(self.init_nodes):
            for i, j in edges.select(init_node, "*"):
                current = j
                tour = [init_node]
                while current not in self.init_nodes:
                    tour.append(current)
                    next = [j for i, j in edges.select(current, "*")][0]
                    current = next
                tour.append(init_node)
                tours.append(tour)

        # TODO:
        if self.come_back_home:
            return tours
        return list(map(lambda x: x[0:-1], tours))

    def get_timestamps(self, m, variables, tour):
        if len(tour) < 2:
            pass

        timestamps = [0]
        edges = m.getAttr("X", variables["x"])
        times = m.getAttr("X", variables["t"])
        finaltime = variables["tf"].getAttr("x")
        for n in tour[1:-1]:
            timestamps.append(times[n])

        timestamps.append(finaltime)
        return timestamps

    def get_cost(self, m):
        return float(m.objVal)

    def solve(
        self,
        tsp,
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        come_back_home: bool = True,
        export_filename: Optional[str] = None,
    ) -> Tuple[List, float]:
        tsp = copy.deepcopy(tsp)
        if len(tsp.nodes) < num_agent:
            num_agent = len(tsp.nodes) - 1

        # Argument Priority: num_agent < init_nodes
        # If both are provided, init_nodes is prioritized
        if init_nodes is None:
            init_nodes = [tsp.nodes[0]] * num_agent
        else:
            num_agent = len(init_nodes)

        self.agents = list(range(num_agent))
        self.init_nodes = init_nodes
        self.come_back_home: bool = come_back_home

        # If we are not coming back homes, we ignore the last traveling cost.
        if not self.come_back_home:
            for init_node in init_nodes:
                for i in set(tsp.nodes) - set([init_node]):
                    tsp.costs[i][init_node] = 0

        # Initialize the Problem
        # Users can add variables m.addVars() and append to variables.update({}) if wanted.
        m, variables = self.initialize_problem(tsp)

        # Users can add their own function to add more constraints
        # Get the variable
        x = variables["x"]
        t = variables["t"]
        tT = variables["tT"]
        tf = variables["tf"]
        non_init_nodes = list(set(tsp.nodes) - set(self.init_nodes))

        for nodes in tsp.nodesets:
            K = sum(init_node in nodes for init_node in self.init_nodes)
            # If the node set includes more initial nodes,
            # then set the #incoming/outgoing edge to the #init_nodes in that set (K)
            # If not exist, just set #in/outgoing edges to 1
            if K == 0:
                K = 1
            # Incoming and Outgoing FLow must be equal at these edges
            m.addConstr(gp.quicksum(x.sum("*", n) for n in nodes) == K, "incoming")
            m.addConstr(gp.quicksum(x.sum(n, "*") for n in nodes) == K, "outgoing")

        # The flow conservation must be enforced to aaaaalllll nodes.
        m.addConstrs((x.sum("*", n) == x.sum(n, "*") for n in tsp.nodes), "flow")

        # Time Window Constraints
        m.addConstrs(
            (
                t[n] >= tsp.tpo.global_constraints[n]["lb"]
                for n in tsp.tpo.global_constraints.keys()
            ),
            "nodeLB",
        )
        m.addConstrs(
            (
                t[n] <= tsp.tpo.global_constraints[n]["ub"]
                for n in tsp.tpo.global_constraints.keys()
            ),
            "nodeUB",
        )
        # Precedent Constraints
        local_const_edges = [
            (src, tgt)
            for src, d in tsp.tpo.local_constraints.items()
            for tgt, b in d.items()
        ]
        m.addConstrs(
            (
                t[tgt] - t[src] >= tsp.tpo.local_constraints[src][tgt]["lb"]
                for (src, tgt) in local_const_edges
            ),
            "edgeLB",
        )
        m.addConstrs(
            (
                t[tgt] - t[src] <= tsp.tpo.local_constraints[src][tgt]["ub"]
                for (src, tgt) in local_const_edges
            ),
            "edgeUB",
        )
        # Time Delays by taking an edge i,j
        m.addConstrs(
            (
                (x[(i, j)] == 1) >> (t[j] - t[i] >= tsp.costs[i][j])
                for (i, j) in tsp.edges
                if j not in self.init_nodes
            ),
            "delay",
        )

        for ii, I in enumerate(self.init_nodes):
            m.addConstrs(
                (
                    (x[(i, I)] == 1) >> (tT[(ii, I)] - t[i] >= tsp.costs[i][I])
                    for i in non_init_nodes
                ),
                "delayTerm",
            )

        # Users can add a new objective by replacing get_edge_cost_objective
        # tf = max([tT1, tT2, ..., tT_|init_nodes|])
        m.addGenConstrMax(tf, tT)
        self.optimize(m, variables, tf)
        if export_filename:
            m.write(export_filename)

        if m.status != GRB.OPTIMAL:
            print("Tour: [], Cost: n/a")
            return [], -1, []

        for var in m.getVars():
            if abs(var.x) > 1e-6:
                print("{0}: {1}".format(var.varName, var.x))
        print("Total matching score: {0}".format(m.objVal))

        tours = self.get_tours(m, variables)
        timestamps = [self.get_timestamps(m, variables, tour) for tour in tours]
        cost = self.get_cost(m)

        return tours, cost, timestamps


if __name__ == "__main__":
    # Number of locations
    n = 4
    nodes = [0, 1, 2, 3]
    # Ready time
    a = [0, 5, 0, 8]
    # Due time
    b = [100, 16, 10, 14]
    # Travel time
    costs: List[List[float]] = [
        [0, 3, 4, 5],
        [3, 0, 5, 4],
        [4, 5, 0, 3],
        [5, 4, 3, 0],
    ]
    # 1. Just Test with the cost (TSP Solver)
    tsp = TSP(nodes, costs)
    solver = MILPTSPSolver()
    tour, cost = solver.solve(tsp)
    print("-" * 100)
    print(tour, cost)
    print("-" * 100)
