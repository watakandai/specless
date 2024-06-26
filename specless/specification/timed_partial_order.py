import itertools
import queue
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import networkx as nx
import numpy as np
import pandas as pd

from specless.typing import TimedTrace

from .partial_order import (
    PartialOrder,
    generate_random_partial_order,
)


class TimedPartialOrder(PartialOrder):
    """Timed Partial Order Model"""

    def __init__(self) -> None:
        super().__init__()
        self.global_constraints: Dict[int, Dict[str, float]] = defaultdict(lambda: {})
        self.local_constraints: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict()
        )
        self.reverse_constraints: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict()
        )

    @classmethod
    def from_csv(cls, path: str) -> Type["TimedPartialOrder"]:
        raise NotImplementedError()

    @classmethod
    def from_constraints(
        cls,
        global_constraints: Dict[int, Tuple[float, float]] = {},
        local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = {},
    ) -> Type["TimedPartialOrder"]:
        """
        Args:
            local_constraints: Forward Timing Constraints. (i, i+1) => (lb, ub)

        For example:
            local_constraints = {
                (1, 2): (5, 10)
                (1, 3): (5, 15)
                (3, 4): (0, 5)
                ...
            }
            tpo = TPO(local_constraints)
        """
        tpo = cls()
        # Convert global constraints to type Dict[int, Dict[str, float]]
        for node, (lb, ub) in global_constraints.items():
            tpo.add_global_constraint(node, lb, ub)
        # Convert local constraints to type Dict[int, Dict[int, Dict[str, float]]]
        for (src, tgt), (lb, ub) in local_constraints.items():
            tpo.add_local_constraint(src, tgt, lb, ub)
        # For the sake of creating edges
        tpo.transitive_reduction()
        return tpo

    def transitive_reduction(self):
        """Transitive Reduce the model to remove redundant edges"""
        g = nx.transitive_reduction(self)
        self.__dict__.update(g.__dict__)

    def add_global_constraint(
        self, node, lb: Optional[float], ub: Optional[float]
    ) -> None:
        """Add a global Constraint to a dict AND as a node"""
        if not isinstance(lb, (int, float)) or lb < 0:
            lb = 0
        if not isinstance(ub, (int, float)) or ub < 0:
            ub = float("inf")
        if ub < lb:
            raise Exception("Upper bound must be greater than the lower bound")

        self.global_constraints[node] = {"lb": lb, "ub": ub}
        super().add_node(node, lb=lb, ub=ub)

    def add_local_constraint(
        self, src_node, tgt_node, lb: Optional[float], ub: Optional[float]
    ) -> None:
        """Add a local Constraint to a dict AND as a node"""
        if not isinstance(lb, (int, float)) or lb < 0:
            lb = 0
        if not isinstance(ub, (int, float)) or ub < 0:
            ub = float("inf")
        if ub < lb:
            raise Exception("Upper bound must be greater than the lower bound")
        super().add_edge(src_node, tgt_node, lb=lb, ub=ub)
        self.local_constraints[src_node][tgt_node] = {"lb": lb, "ub": ub}
        self.reverse_constraints[tgt_node][src_node] = {"lb": lb, "ub": ub}

    def satisfy(self, demonstration: TimedTrace) -> bool:
        """Checks if the timed_trace satisfies the TPO constraints

        Args:
            demonstration (List[Tuple[int, float]]): A list of event[int] and timestamps[float]

        Returns:
            bool: True if satisfies the TPO or else False
        """
        # timed_trace must contain all events of the TPO
        events = list(list(zip(*demonstration))[0])

        # TODO: Add nodes.
        # if set(events) != set(self.nodes()):
        #     return False

        event_to_time = {e: t for (e, t) in demonstration}

        for tgt, tgt_time in demonstration:
            # Node Constraint
            if tgt in self.global_constraints:
                bound = self.global_constraints[tgt]
                if tgt_time < bound["lb"] or bound["ub"] < tgt_time:
                    return False

            # Edge Constraint
            if tgt in self.reverse_constraints:
                # Search if there's any violation
                for src, bound in self.reverse_constraints[tgt].items():
                    src_time = event_to_time[src]
                    dt = tgt_time - src_time
                    if dt < bound["lb"] or bound["ub"] < dt:
                        print(
                            f"Edge {src}->{tgt} for dt={dt} did not satisfy {bound['lb']} <= dt <= {bound['ub']}"
                        )
                        return False
        return True

    def penalize(self, tour) -> float:
        """Penelize the tour if the tour violates the constraints

        Args:
            tour (_type_): _description_

        Returns:
            float: _description_
        """
        if len(tour) == 0:
            return 0

        makespan, times = self.makespan2(tour, returnTimes=True)
        if makespan < 0:
            makespan, times = self.makespan(tour, returnTimes=True)
        penalty = 0

        for curr_node in tour:
            curr_time = times[curr_node]

            # Node Time Constraint UB
            if (
                curr_node in self.global_constraints
                and curr_time > self.global_constraints[curr_node]["ub"]
            ):
                penalty += curr_time - self.global_constraints[curr_node]["ub"]
            # Edge Time Constraint UB
            for src_node, bound in self.reverse_constraints[curr_node].items():
                src_time = times[src_node]
                diff = curr_time - src_time
                if diff > bound["ub"]:
                    penalty += diff - bound["ub"]

        return penalty

    def modify_cost(
        self, matrix: Dict[int, Dict[int, float]]
    ) -> Dict[int, Dict[int, float]]:
        # For i < j, dist[j, i] = inf
        reverse_constraints = self.get_reverse_constraints()
        for tgt_node, src_nodes in reverse_constraints.items():
            for src_node in src_nodes:
                del matrix[tgt_node][src_node]
        return matrix

    def makespan(self, tour, returnTimes: bool = False):
        """Penelize the tour if the tour violates the constraints

        Args:
            tour (_type_): _description_

        Returns:
            float: _description_
        """
        if len(tour) == 0:
            return 0
        curr_time = 0
        idx = 0
        if idx in self.global_constraints:
            curr_time = self.global_constraints[idx].get("lb")
        prev_node = tour[idx]
        times: Dict = {idx: curr_time}

        idx += 1

        for curr_node in tour[idx:]:
            curr_time += TSP.dist(prev_node, curr_node)

            ### Increment Time if arrived too early

            # Node Time Constraint LB
            # Check if curr_node have the lb. If not,
            if curr_node in self.global_constraints:
                lb = self.global_constraints[curr_node].get("lb")
                curr_time = curr_time if lb is None else max(curr_time, lb)
            # Edge Time Constraint LB
            for src_node, bound in self.reverse_constraints[curr_node].items():
                if src_node in times:
                    curr_time = max(curr_time, times[src_node] + bound["lb"])

            ### Penalize if the time goes beyond the upperbound

            # Check if the precedent constraints are satisfied
            prev_times = [
                times[src_node] + TSP.dist(src_node, curr_node)
                for src_node in self.predecessors(curr_node)
                if src_node in times
            ]
            if len(prev_times) > 0:
                max_prev_time = max(prev_times)
                if curr_time < max_prev_time:
                    curr_time = times[src_node]

            # Edge Time Constraint UB
            times[curr_node] = curr_time
            prev_node = curr_node
            idx += 1

        curr_time += TSP.dist(tour[-1], tour[0])

        if returnTimes:
            return curr_time, times
        return curr_time

    def makespan2(self, tour, returnTimes: bool = False):
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as m:
                # Create continuous time variables [1, n]
                t = m.addVars(self.nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="times")
                tf = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="tFinal")

                m.addConstrs(
                    (t[i] + TSP.edges[i][j] <= t[j] for i, j in zip(tour, tour[1:])),
                    "tour",
                )
                # Time Window Constraints
                m.addConstrs(
                    (
                        t[n] >= self.global_constraints[n]["lb"]
                        for n in self.global_constraints.keys()
                    ),
                    "nodeLB",
                )
                m.addConstrs(
                    (
                        t[n] <= self.global_constraints[n]["ub"]
                        for n in self.global_constraints.keys()
                    ),
                    "nodeUB",
                )
                # Precedent Constraints
                local_const_edges = [
                    (src, tgt)
                    for src, d in self.local_constraints.items()
                    for tgt, b in d.items()
                ]
                m.addConstrs(
                    (
                        t[tgt] - t[src] >= self.local_constraints[src][tgt]["lb"]
                        for (src, tgt) in local_const_edges
                    ),
                    "edgeLB",
                )
                m.addConstrs(
                    (
                        t[tgt] - t[src] <= self.local_constraints[src][tgt]["ub"]
                        for (src, tgt) in local_const_edges
                    ),
                    "edgeUB",
                )

                # Users can add a new objective by replacing get_edge_cost_objective
                # tf = max([tT1, tT2, ..., tT_|init_nodes|])
                m.addGenConstrMax(tf, t)
                m.setObjective(tf, GRB.MINIMIZE)

                # save model for inspection
                class_name = type(self).__name__
                m.write(f"{class_name}.lp")
                m.optimize()

                if m.status != GRB.OPTIMAL:
                    return -1, []

                times = {}
                for var in m.getVars():
                    if "times" in var.varName:
                        pattern = r"times\[(\d+)\]"
                        matches = re.findall(pattern, var.varName)
                        ns = [int(match) for match in matches]
                        if len(ns) == 1:
                            times[ns[0]] = var.x

                makespan = float(m.objVal)
                if returnTimes:
                    return makespan, times

                return makespan

    def print(self, tour):
        _, times = self.makespan(tour, returnTimes=True)

        for n in tour:
            t = times[n]
            print(f"Node={n}, Time={t}")
            if n in self.global_constraints:
                glb = self.global_constraints[n]["lb"]
                gub = self.global_constraints[n]["ub"]
                print(f"\t{glb} <= {t} <= {gub}")

            for src, bound in self.reverse_constraints[n].items():
                ts = times[src]
                llb = bound["lb"]
                lub = bound["ub"]
                print(f"\t{llb} <= n{n}(t={t}) - n{src}(t={ts}) <= {lub}")

    def __str__(self):
        string = ""
        for n, d in self.global_constraints.items():
            lb = d["lb"]
            ub = d["ub"]
            string += f"\t{lb} <= t_{n} <= {ub}\n"

        for src, dd in self.local_constraints.items():
            for tgt, d in dd.items():
                lb = d["lb"]
                ub = d["ub"]
                string += f"\t{lb} <= t_{tgt} - t_{src} <= {ub}\n"
        return string


class Service:
    name: str
    type: str
    from_str: str
    to_str: str
    service_time: float
    controllabe: bool
    precedences: List

    def __init__(
        self,
        name,
        type,
        from_str,
        to_str,
        service_time,
        controllable,
        precedences: List,
    ):
        self.name = name
        self.type = type
        self.from_str = from_str
        self.to_str = to_str
        self.service_time = service_time
        self.controllable = controllable
        self.precedences = precedences

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class ServiceTimedPartialOrder(TimedPartialOrder):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def load_services(filepath: str) -> List[Service]:
        """Parse Timed Partial Order with Service Time from CSV
        CSV Columns: No.,Activity,ServiceTime,Service,From,To,Precedence,Operation

        Arguments
        =========
        filepath: str
            Path to the CSV file

        Returns
        =======
        ServiceTimedPartialOrder

        """

        df = pd.read_csv(filepath)

        services = []
        for index, row in df.iterrows():
            name = row["Activity"]
            type = row["Service"]
            from_str = row["From"]
            to_str = row["To"]
            service_time = row["ServiceTime"]
            if not isinstance(from_str, str):
                from_str = ""
            if not isinstance(to_str, str):
                to_str = ""
            controllable = type != "Uncontrollable"
            precedences = [
                df[df["No."] == int(n)]["Activity"].values[0]
                for n in df[df["No."] == row["No."]]["Precedence"].values[0].split(",")
                if n != " "
            ]
            services.append(
                Service(
                    name,
                    type,
                    from_str,
                    to_str,
                    service_time,
                    controllable,
                    precedences,
                )
            )
        return services

    @classmethod
    def from_services(cls, services):
        pass


def generate_random_timed_partial_order(
    num_nodes: int,
    edge_probability: float = 0.3,
    global_clock_probability: float = 0.3,
    local_clock_probability: float = 0.3,
    fixed_time_window: int = 30,
) -> TimedPartialOrder:
    # Generate a random PO
    po: PartialOrder = generate_random_partial_order(num_nodes, edge_probability)

    assert len(po.nodes) == num_nodes, po.nodes

    # Goal is to fill in these constraints such that the TPO is valid
    global_constraints: Dict[int, Tuple[float, float]] = {}
    local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = {
        e: (0.0, float("inf")) for e in po.edges()
    }
    # Mapping from curr_node -> List[prev_node]
    reverse_constraints = po.get_reverse_constraints()

    # Initial Nodes
    initial_nodes = [x for x in po.nodes() if po.in_degree(x) == 0]
    # Start with the initial nodes
    search_queue: queue.Queue = queue.Queue()
    for n in initial_nodes:
        search_queue.put(n)
    visited = set()

    # Must keep track of the feasible bounds (based on the global clock) at each node
    feasible_global_bounds: Dict = {}  # Node -> bounds

    while not search_queue.empty():
        # Get a Current Node
        curr_node = search_queue.get()
        # Predecessor's Upper Bounds
        ubs = [feasible_global_bounds[n]["ub"] for n in po.predecessors(curr_node)]
        # The worst case UB becomes the current node's LB
        G_LB = 0 if len(ubs) == 0 else np.max(ubs)

        # Geenrate a "underlying" global LB and UB (H_G_LB/UB) for the current node.
        # G_LB =========== H_G_LB ----------- H_G_UB =========== G_UB
        # All the global and local bounds must be in between ===== regions.
        G_UB = G_LB + fixed_time_window
        H_G_LB = random.randint(G_LB, G_LB + fixed_time_window / 2)
        H_G_UB = random.randint(H_G_LB + 1, G_LB + fixed_time_window)

        feas_LB = G_LB
        feas_UB = G_UB

        # Assign a global time constraint with some probability
        if random.random() < global_clock_probability:
            global_lb = random.randint(G_LB, H_G_LB)
            global_ub = random.randint(H_G_UB, G_UB)
            feas_LB = max(feas_LB, global_lb)
            feas_UB = min(feas_UB, global_ub)
            if global_lb > global_ub:
                raise Exception("Upper bound must be greater than the lower bound")
            global_constraints[curr_node] = (global_lb, global_ub)

        # Assign a local time constraint with some probability
        for prev_node in reverse_constraints[curr_node]:
            if random.random() < local_clock_probability:
                # lb_i <= ti <= ub_i
                bound_i = feasible_global_bounds[prev_node]
                # lb_j <= tj <= ub_j
                lb_j = random.randint(G_LB, H_G_LB)
                ub_j = random.randint(H_G_UB, G_UB)
                feas_LB = max(feas_LB, lb_j)
                feas_UB = min(feas_UB, ub_j)
                # -ub_i <= -ti <= -lb_i
                # lb_{ij} <= tj - ti <= ub_{ij}
                # lb_j - ub_i <= tj - ti <= ub_j - lb_i
                local_lb = lb_j - bound_i["ub"]
                local_ub = ub_j - bound_i["lb"]
                if local_lb > local_ub:
                    raise Exception("Upper bound must be greater than the lower bound")
                local_constraints[(prev_node, curr_node)] = (local_lb, local_ub)

        if feas_LB > feas_UB:
            raise Exception("Upper bound must be greater than the lower bound")
        feasible_global_bounds[curr_node] = {"lb": feas_LB, "ub": feas_UB}

        # Mark the current node as visited
        visited.add(curr_node)
        # Now, add its children to the search queue
        for next_node in po.successors(curr_node):
            # Only append if its parents are all visited
            all_visited = [n in visited for n in po.predecessors(next_node)]
            if all(all_visited):
                search_queue.put(next_node)

    tpo = TimedPartialOrder.from_constraints(global_constraints, local_constraints)

    assert len(tpo.nodes) == num_nodes, tpo.nodes
    return tpo


def fixed_time_gap():
    return 1


def generate_random_constraints(
    nodes: List[int],
    initial_nodes: List[int],
    edge_costs: List[List[float]],
    num_constraint: Union[int, float],
    node_costs: Optional[List[float]] = None,
    time_gap_callback: Callable = fixed_time_gap,
):
    """Generate random constraints to generate a TPO
    Assume nodes includes the "initial nodes (depot)" which starts from 0.
    """
    if node_costs is None:
        node_costs = defaultdict(lambda: 0)

    num_node = len(nodes)
    num_total_constraint = int(num_node * (num_node - 1) / 2)
    if isinstance(num_constraint, float):
        num_constraint = int(num_constraint * num_total_constraint)
    assert num_constraint <= num_total_constraint

    timed_trace = generate_random_timed_trace(
        nodes, initial_nodes, edge_costs, node_costs
    )
    # #Constraints is n + n*(n-1)/2.  Let n=|V|, Then,
    # For each v\inV, lb <= t_u <= ub. |V| -> n
    # For each u, v \in E, lb <= t_v - t_u <= lb. |E|-> n*(n-1)/2
    timed_trace_dict = {n: t for n, t in timed_trace}
    trace = list(zip(*timed_trace))[0]
    edge_pairs = list(itertools.combinations(trace, 2))
    edges = random.choices(edge_pairs, k=num_constraint)

    tpo: TimedPartialOrder = TimedPartialOrder()
    for u, v in edges:
        u_time = timed_trace_dict[u]
        v_time = timed_trace_dict[v]
        diff = v_time - u_time
        tpo.add_local_constraint(
            u, v, diff - time_gap_callback(), diff + time_gap_callback()
        )

    return tpo


def generate_random_timed_trace(
    nodes,
    initial_nodes: List[int],
    edge_costs: List[List[float]],
    node_costs: Optional[List[float]] = None,
) -> List[Tuple[int, float]]:
    """Generate a random timed trace.
    Assume nodes includes the "initial nodes (depot)" which starts from 0.
    """

    if node_costs is None:
        node_costs = defaultdict(lambda: 0)

    num_agent = len(initial_nodes)
    remaining_nodes = set(nodes) - set(initial_nodes)
    curr_states = {i: (n, 0) for i, n in enumerate(initial_nodes)}
    timed_trace = [(n, 0) for n in initial_nodes]
    timed_traces = {i: [(n, 0)] for i, n in enumerate(initial_nodes)}

    while len(remaining_nodes) != 0:
        index = random.randint(1, num_agent) - 1
        curr_node, curr_time = curr_states[index]

        next_node = random.choice(list(remaining_nodes))
        travel_distance = edge_costs[curr_node][next_node]
        service_distance = node_costs[next_node]
        next_time = curr_time + travel_distance + service_distance

        curr_states[index] = (next_node, next_time)
        remaining_nodes = remaining_nodes - set([next_node])
        timed_trace.append((next_node, next_time))
        timed_traces[index].append((next_node, next_time))

    return sorted(timed_trace, key=lambda t: t[1])
