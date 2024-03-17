from typing import List

import gurobipy as gp
from gurobipy import GRB

from specless.tsp.tsp import TSP, Node
from specless.specification.base import Specification
from specless.factory.tspbuilder import TSPBuilder

# from .base import TSPSolver, TSPWithTPOSolver


class RobustAnalysis:
    def __init__(self):
        pass

    def analyze(self, tsp: TSP, specification: Specification, tours: List[List[int]], outputflag: int=0):

        env = gp.Env()
        env.setParam("OutputFlag", outputflag)

        m = gp.Model(env=env)

        # Create decision variables for choosing edges
        delta_nodes = m.addVars(tsp.nodes, vtype=GRB.CONTINUOUS, name="nodes")
        delta_edges = m.addVars(tsp.edges, vtype=GRB.CONTINUOUS, name="edges")

        # Create continuous time variables [1, n]
        e = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="epsilon")

        # Create continuous time variables
        t = m.addVars(tsp.nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="times")
        # The final time
        tf = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tFinal")

        for tour in tours:
            prev = tour[0]
            for curr in tour[1:-1]:
                dj: float = tsp.services[curr]
                dij: float = tsp.costs[prev][curr]
                m.addConstr(t[curr] == t[prev] + dj + dij + delta_nodes[curr] + delta_edges[(prev, curr)], "edge{prev}-{curr}")
                # m.addConstr(t[curr] == t[prev] + dj + dij + delta_nodes[curr], "node{curr}")
                prev: int = curr

        # Time Window Constraints
        m.addConstrs(
            (
                t[n] >= specification.global_constraints[n]["lb"]
                for n in specification.global_constraints.keys()
            ),
            "nodeLB",
        )
        m.addConstrs(
            (
                t[n] <= specification.global_constraints[n]["ub"]
                for n in specification.global_constraints.keys()
            ),
            "nodeUB",
        )
        # Precedent Constraints
        local_const_edges = [
            (src, tgt)
            for src, d in specification.local_constraints.items()
            for tgt, b in d.items()
        ]
        m.addConstrs(
            (
                t[tgt] - t[src] >= specification.local_constraints[src][tgt]["lb"]
                for (src, tgt) in local_const_edges
            ),
            "edgeLB",
        )
        m.addConstrs(
            (
                t[tgt] - t[src] <= specification.local_constraints[src][tgt]["ub"]
                for (src, tgt) in local_const_edges
            ),
            "edgeUB",
        )
        # m.addConstrs(
        #     (
        #         e <= t[tgt] - t[src]
        #         for (src, tgt) in local_const_edges
        #     ),
        #     "boundUB",
        # )

        # Bound the Robustness Bound
        m.addConstrs((e <= delta_nodes[n] for n in tsp.nodes), "NodeUBOnE")
        m.addConstrs((e <= delta_edges[edge] for edge in tsp.edges), "EdgeUBOnE")
        # m.addConstrs((delta_nodes[n] <= e for n in tsp.nodes), "ub")
        # m.addConstrs((-e <= delta_nodes[n] for n in tsp.nodes), "lb")
        # m.addConstrs((delta_edges[edge] <= e for edge in tsp.edges), "ub")
        # m.addConstrs((-e <= delta_edges[edge] for edge in tsp.edges), "lb")

        # Objective Function
        m.setObjective(e, GRB.MAXIMIZE)

        # Opimize!
        m.optimize()

        if m.status != GRB.OPTIMAL:
            return -1, [], -1

        for var in m.getVars():
            if abs(var.x) > 1e-6:
                print("{0}: {1}".format(var.varName, var.x))

        bound = e.getAttr("x")
        times = m.getAttr("X", t)
        deltas = m.getAttr("X", delta_nodes)
        print(times)
        print(deltas)
        finaltime = tf.getAttr("x")

        return bound, times, finaltime

