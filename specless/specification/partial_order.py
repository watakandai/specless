import queue
from collections import defaultdict

from specless.specification.base import Specification
from specless.typing import Trace


class PartialOrder(Specification):
    """Partial Order Model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def partial_order(self):
        partial_order = defaultdict(lambda: set())
        final_nodes = [n for n in self.nodes() if self.out_degree(n) == 0]
        search_queue = queue.Queue()
        for n in final_nodes:
            search_queue.put(n)
            partial_order[n] = set()

        while not search_queue.empty():
            curr_node = search_queue.get()

            for prev_node in self.predecessors(curr_node):
                # Add successor's sets to previous node's partial order set
                partial_order[prev_node].union(partial_order[curr_node])
                # Add the successor as well
                partial_order[prev_node].union([curr_node])

                search_queue.put(prev_node)

        return partial_order

    def satisfy(self, demonstration: Trace) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (Trace): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        nexts = [n for n in self.nodes if self.in_degree(n) == 0]
        visited = []
        for node in demonstration:
            if node not in nexts:
                return False

            visited.append(node)
            nexts += [
                tgt
                for tgt in self.successors(node)
                if all([src in visited for src in self.predecessors(tgt)])
            ]
            nexts.remove(node)
        return True

    def get_reverse_constraints(self):
        reverse_constraints = defaultdict(lambda: set())
        search_queue = queue.Queue()
        visited = set()
        initial_nodes = [x for x in self.nodes() if self.in_degree(x) == 0]
        for n in initial_nodes:
            search_queue.put(n)

        while not search_queue.empty():
            curr_node = search_queue.get()
            for prev_node in self.predecessors(curr_node):
                reverse_constraints[curr_node].add(prev_node)
                reverse_constraints[curr_node] |= reverse_constraints[prev_node]

            for next_node in self.successors(curr_node):
                if next_node not in visited:
                    search_queue.put(next_node)
                    visited.add(next_node)

        return reverse_constraints


# Function to generate a random partial order (DAG)
def generate_random_partial_order(num_nodes, probability=0.3):
    G = PartialOrder()
    # Add nodes
    G.add_nodes_from(list(range(num_nodes)))
    # Add directed edges randomly with a given probability
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < probability:
                G.add_edge(i, j)

    for n in range(num_nodes):
        if G.in_degree(n) == 0 and G.out_degree(n) == 0:
            if random.random() < n / (num_nodes - 1):
                src = random.choice(list(range(0, n)))
                G.add_edge(src, n)
            else:
                tgt = random.choice(list(range(n + 1, num_nodes)))
                G.add_edge(n, tgt)
        assert G.in_degree(n) != 0 or G.out_degree(n) != 0
    return G
