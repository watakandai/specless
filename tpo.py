import copy
import queue
import random
import itertools
import numpy as np
import networkx as nx
from networkx.algorithms.dag import transitive_reduction
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union
from tpossp import utils


class TPO(nx.DiGraph):
    """Timed Partial Order Model
    """
    global_constraints: Dict[int, Dict[str, float]]
    local_constraints: Dict[int, Dict[int, Dict[str, float]]]
    reverse_constraints: Dict[int, Dict[int, Dict[str, float]]]

    def __init__(self, global_constraints: Dict[int, Tuple[float, float]]={},
                       local_constraints: Dict[Tuple[int, int], Tuple[float, float]]={}):
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
        # Convert global constraints to type Dict[int, Dict[str, float]]
        global_constraints_ = {}
        for node, bound in global_constraints.items():
            if not isinstance(bound[0], (int, float)) or bound[0]<0:
                bound = (0, bound[1])
            if not isinstance(bound[1], (int, float)) or bound[1]<0:
                bound = (bound[0], float('inf'))
            if bound[1] < bound[0]:
                raise Exception('Upper bound must be greater than the lower bound')
            global_constraints_[node] = {'lb': bound[0], 'ub': bound[1]}
        self.global_constraints = global_constraints_

        # Convert local constraints to type Dict[int, Dict[int, Dict[str, float]]]
        # ex.) {Node U: {Node V: {LB: 0, UB: 10}}}
        local_constraints_ = defaultdict(lambda: {})
        for k, bound in local_constraints.items():
            if not isinstance(bound[0], (int, float)) or bound[0]<0:
                bound = (0, bound[1])
            if not isinstance(bound[1], (int, float)) or bound[1]<0:
                bound = (bound[0], float('inf'))
            if bound[1] < bound[0]:
                raise Exception('Upper bound must be greater than the lower bound')
            local_constraints_[k[0]][k[1]] = {'lb': bound[0], 'ub': bound[1]}
        self.local_constraints = dict(local_constraints_)

        # Store Reverse Constraints for the "satisfy" function to easily access the constraint
        # ex.) {Node V: {Node U: {LB: 0, UB: 10}}
        reverse_constraints_ = defaultdict(lambda: {})
        for src, d in self.local_constraints.items():
            for tgt, bound in d.items():
                reverse_constraints_[tgt][src] = bound
        self.reverse_constraints = reverse_constraints_

        # For the sake of creating edges
        super().__init__(dict(local_constraints_))

        for node_id, bound in self.global_constraints.items():
            self.add_node(node_id, lb=bound['lb'], ub=bound['ub'])
        for src, d in self.local_constraints.items():
            for tgt, bound in d.items():
                self.add_edge(src, tgt, lb=bound['lb'], ub=bound['ub'])

        # Reduce the redundant edges
        g = nx.transitive_reduction(self)
        self.__dict__.update(g.__dict__)

    def satisfy(self, timed_trace: List[Tuple[int, float]], print_reason: bool=False) -> bool:
        """Checks if the timed_trace satisfies the TPO constraints

        Args:
            timed_trace (List[Tuple[int, float]]): A list of event[int] and timestamps[float]

        Returns:
            bool: True if satisfies the TPO or else False
        """
        # timed_trace must contain all events of the TPO
        events = list(list(zip(*timed_trace))[0])

        # TODO: Add nodes.
        # if set(events) != set(self.nodes()):
        #     return False

        event_to_time = {e: t for (e, t) in timed_trace}

        for (tgt, tgt_time) in timed_trace:
            # Node Constraint
            if tgt in self.global_constraints:
                bound = self.global_constraints[tgt]
                if tgt_time < bound['lb'] or bound['ub'] < tgt_time:
                    if print_reason:
                        print(f"Node {tgt} at {tgt_time} did not satisfy {bound['lb']} <= t <= {bound['ub']}")
                    return False

            # Edge Constraint
            if tgt in self.reverse_constraints:
                # Search if there's any violation
                for src, bound in self.reverse_constraints[tgt].items():
                    src_time = event_to_time[src]
                    dt = tgt_time - src_time
                    if dt < bound['lb'] or bound['ub'] < dt:
                        print(f"Edge {src}->{tgt} for dt={dt} did not satisfy {bound['lb']} <= dt <= {bound['ub']}")
                        return False
        return True

    def print(self):
        print("Global Constraints")
        print('-'*50)
        spacing = 2
        ss = f'<{spacing}'
        for node_id, bound in self.global_constraints.items():
            print(f"{bound['lb']:<5} <= t{node_id:{ss}} <= {bound['ub']}")

        print("\nLocal Constraints")
        print('-'*50)
        for src, d in self.local_constraints.items():
            for tgt, bound in d.items():
                print(f"{bound['lb']:<5} <= t{tgt:{ss}} - t{src:<2} <= {bound['ub']}")

    def draw(self, filepath):
        node_label_function = lambda n, data: utils.node_label_function(n, data, ['lb', 'ub'])
        edge_label_function = lambda u, v, data: utils.edge_label_function(u, v, data, ['lb', 'ub'])
        utils.add_labels(self, node_label_function, edge_label_function)
        utils.draw(self, filepath)

    # TODO: Convert locals to globals.
    def get_all_global_constraints(self):
        """Slowly progress based on the TPO topology.
        """
        # global_constraints = dict(defaultdict(lambda: {'lb': float('-inf'), 'ub': float('inf')}))
        # for n, b in self.global_constraints.items():
        #     global_constraints[n] = b
        global_constraints = copy.deepcopy(self.global_constraints)

        initial_nodes = [n for n in self.nodes
                         if len(list(self.predecessors(n)))==0 and
                         n in self.global_constraints]
        search_queue = queue.Queue()
        for n in initial_nodes:
            search_queue.put(n)

        while not search_queue.empty():

            src = search_queue.get()

            if src not in global_constraints:
                lb = float('-inf')
                ub = float('inf')
                for p, localb in self.reverse_constraints[src].items():
                    global_lb = global_constraints[p]['lb']
                    global_ub = global_constraints[p]['ub']
                    local_lb = localb['lb']
                    local_ub = localb['ub']
                    lb = max(lb, global_lb + local_lb)
                    ub = min(ub, global_ub + local_ub)
                global_constraints[src] = {'lb': lb, 'ub': ub}

            for tgt in self.successors(src):
                if all(n in global_constraints for n in self.predecessors(tgt)):
                    search_queue.put(tgt)

        return global_constraints

    @staticmethod
    def generate_random_constraints(nodes, edge_costs, node_costs,
                                    initial_nodes, nodesets,
                                    num_constraint: Union[int, float], use_min=True,
                                    epsilon: int=0.0):
        """Generate random constraints to generate a TPO
        Assume nodes includes the "initial nodes (depot)" which starts from 0.
        """
        num_node = len(nodes)
        num_total_constraint = int(num_node * (num_node - 1) / 2)
        if isinstance(num_constraint, float):
            num_constraint = int(num_constraint*num_total_constraint)
        assert num_constraint  <= num_total_constraint

        timed_trace = TPO.generate_random_timed_trace(nodes, edge_costs, node_costs,
                                                      initial_nodes, nodesets)
        # #Constraints is n + n*(n-1)/2.  Let n=|V|, Then,
        # For each v\inV, lb <= t_u <= ub. |V| -> n
        # For each u, v \in E, lb <= t_v - t_u <= lb. |E|-> n*(n-1)/2
        timed_trace_dict = {n: t for n, t in timed_trace}
        print(list(zip(*timed_trace)))
        trace = list(zip(*timed_trace))[0]
        edge_pairs = list(itertools.combinations(trace, 2))
        edges = random.choices(edge_pairs, k=num_constraint)

        # gamma distribution: mean=k\theta, var=k(\theta)^2
        # k, \theta
        # heuristics: let's set the mean to be 2*E_n[shortestpath(n)]

        scale = 2.0
        if use_min:
            k = np.mean([min(edge_costs[n]) for n in nodes]) / scale
        else:
            k = np.mean(edge_costs) / scale

        global_constraints, local_constraints = {}, {}
        for (u, v) in edges:
            u_time = timed_trace_dict[u]
            v_time = timed_trace_dict[v]
            diff = v_time - u_time
            # s = np.random.gamma(k, scale, 2)
            # u_ind = trace.index(u)
            # v_ind = trace.index(v)
            # stp = v_ind - u_ind
            local_constraints[(u, v)] = (diff-epsilon, diff+epsilon)
            local_constraints[(u, v)] = (diff-epsilon, diff+epsilon)

        return global_constraints, local_constraints

    @staticmethod
    def generate_random_timed_trace(nodes, edge_costs, node_costs,
                                    initial_nodes, nodesets):
        """Generate a random timed trace.
        Assume nodes includes the "initial nodes (depot)" which starts from 0.
        """
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

        # for i, tt in timed_traces.items():
        #     print(f"Agent{i}: TimedTrace={tt}")

        return sorted(timed_trace, key=lambda t: t[1])


if __name__ == '__main__':

    global_constraints = {1: (0, 5),
                          6: (20, 60)}
    local_constraints = {(1, 3): (10, None),
                         (1, 5): (0, 15),
                         (4, 5): (5, None),
                         (4, 6): (4, 10),
                         (5, 6): (0, 8),
                         }

    tpo = TPO(global_constraints, local_constraints)

    satisfying_timed_trace = [(1, 2), (2, 10), (3, 12), (4, 12), (5, 17), (6, 20)]
    b = tpo.satisfy(satisfying_timed_trace)
    assert b

    unsatifying_timed_trace = [(1, 3), (2, 10), (3, 15), (4, 16), (5, 21), (6, 30)]
    b = tpo.satisfy(unsatifying_timed_trace)
    assert not b
