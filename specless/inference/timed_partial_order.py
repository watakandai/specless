"""
Inference Algorithm
===================
Inference algorithms then use such demonstrations to come up with a specification.
>> import specless as sl
>> traces = [[a,b,c], [a,b,b,c], [a,a,b,b,c]]
>> dataset = sl.ArrayDataset(traces)
>> inference = sl.TPOInference()
>> specification = inference.infer(demonstrations)
"""
import copy
import queue
import random
from collections import defaultdict
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import cvxopt as cvx
import networkx as nx
import numpy as np

from specless.dataset import ArrayDataset, BaseDataset
from specless.inference.base import InferenceAlgorithm
from specless.inference.partial_order import POInferenceAlgorithm
from specless.specification.base import Specification
from specless.specification.partial_order import PartialOrder
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.typing import (
    EdgeBoundDict,
    NodeBoundDict,
    Symbol,
    TimedTraceList,
    TimeStamp,
)


class TPOInferenceAlgorithm(InferenceAlgorithm):
    """The inference algorithm for inferring a TPO from a list of TimedTraces.

    Args:
        InferenceAlgorithm (_type_): _description_
    """

    def __init__(
        self, heuristic: str = "order", decimals: int = 2, threshold: float = 0.5
    ):
        super().__init__()
        self.heuristic = heuristic
        self.decimals = decimals
        self.threshold = threshold
        self.graph = None
        self.partial_order = None

    def infer(self, dataset: BaseDataset) -> Union[Specification, Exception]:
        """Infer a Timed Partial Order (TPO) from a list of timed traces

        Implementation in detail:
            1. For each symbol, we keep all possible
                - forward constraints,
                    ex.) symbol < next_symbol,
                - backward constraints,
                    ex.) prev_symbol < symbol.

            2. If there is a hard constraint in the order,
            there should NEVER be a same symbol in
            forward constraints and backwards constraints.
            Thus,
                linear constraints = forward_constraints - backward_constraints.

            3. We construct a graph based on the linear constraints.

        Args:
            dataset (Dataset):        Timed Trace Data

        Raises:
            NotImplementedError: _description_

        Returns:
            Specification:                  Timed Partial Order
        """
        sorted_dataset: BaseDataset = copy.deepcopy(dataset)
        sorted_dataset.apply(
            lambda data: data.sort_values(by="timestamp", inplace=True)
        )
        traces: List = dataset.tolist(key="symbol")
        timedtraces: List = dataset.tolist()

        # Find a partial order
        inference = POInferenceAlgorithm()
        partial_order: Dict = inference.get_partial_order(traces)
        po: Specification = inference.infer(sorted_dataset)
        # Infer Timing Constraints
        global_constraints, local_constraints = self.infer_time_constraints(
            timedtraces, po, partial_order
        )

        return TimedPartialOrder.from_constraints(global_constraints, local_constraints)

    def infer_time_constraints(
        self,
        traces: TimedTraceList,
        po: PartialOrder,
        partial_order: Dict[str, List[str]],
        debug: bool = False,
        decimals: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple:
        """
        Infer Time Bounds on Nodes and Edges given Partial Order.

        Optimization Problem:
            min         c x
            s.t.
                A x <= b

            variables:
                tau_ei:     Time Variable at Event i
            values:
                t_ei:       Time at Event i from data
                T_ei:       Times at Event i for all traces, i.e., T_ei = [t_ei, ...]

            min:        \sum (u_eij - l_eij) + \sum (u_ei - l_eij)
            s.t.
                tau_e1 >= max(0, min(T_e1))                 => -tau_e1 <= -min
                tau_e1 <= max(T_e1)
                    ...
                tau_e2 - tau_e1 >= max(0, min(T_e2-T_e1))   => -(tau_e2 - tau_e1) <= -min
                tau_e2 - tau_e1 <= max(T_e2-T_e1)
                    ...

        Thus,
            num_variable = num_event
            num_constraint = 2*num_event + 2*num_pair

            x.shape = (num_variable, 1) or simply vector x
            c.shape = (1, num_variable) or simply vector c^T
            A.shape = ((num_constraint) x (num_variable))
            b.shape = (num_constraint, 1) or simply vector b

        """
        if decimals is None:
            decimals = self.decimals

        if threshold is None:
            threshold = self.threshold

        no_order_at_all = all([len(tgts) == 0 for src, tgts in partial_order.items()])
        if no_order_at_all:
            return {}, {}

        lp = TimeConstraintsLP(traces, partial_order, decimals, threshold)
        next_edge_iter = self.select_next_edge_iterator(lp, po, partial_order)
        post_processing_func = self.select_post_processing_func()

        minimal_event_pair_to_bound = {}
        edge = ("", "")

        while edge is not None:
            # An edge eis selected based on the selected heuristic
            edge = next_edge_iter()
            if edge is None:
                continue

            source_event, target_event = edge

            redundant = False

            # Check for the Lower Bound
            sol = lp.solvefor(source_event, target_event, lb=True, debug=debug)
            orig_lb = sol["orig_bound"]
            if sol["redundant"]:
                redundant = True

            # Check for the Upper Bound
            sol = lp.solvefor(source_event, target_event, lb=False, debug=debug)
            orig_ub = sol["orig_bound"]
            if sol["redundant"]:
                redundant = True

            if redundant:
                if debug:
                    print(
                        "Removing Constraint ...",
                        lp.get_events_string(source_event, target_event),
                    )
                lp.remove_constraint(source_event, target_event, lb=True)
                lp.remove_constraint(source_event, target_event, lb=False)
            else:
                minimal_event_pair_to_bound[(source_event, target_event)] = (
                    orig_lb,
                    orig_ub,
                )

            # Post processing needed for some heuristics to put the constraints back
            post_processing_func(
                lp, source_event, target_event, redundant, orig_lb, orig_ub
            )

        return lp.event_to_bound, minimal_event_pair_to_bound

    def select_next_edge_iterator(
        self,
        lp: Type["TimeConstraintsLP"],
        po: PartialOrder,
        partial_order: Dict[str, List[str]],
    ) -> Callable:
        """Select next_edge_iter function"""

        pairs = copy.deepcopy(lp.pairs)

        if self.heuristic == "random":
            random.shuffle(pairs)
        elif self.heuristic == "order":
            pass
        elif self.heuristic == "distant":
            g = nx.transitive_reduction(self.graph)
            pair_to_dist: Dict[tuple, Any] = {
                (s, t): nx.shortest_path_length(g, s, t) for s, t in pairs
            }
            pairs: List[Any] = [
                p for p in sorted(pair_to_dist, key=pair_to_dist.get, reverse=True)
            ]
        elif self.heuristic == "near":
            g = nx.transitive_reduction(self.graph)
            pair_to_dist = {(s, t): nx.shortest_path_length(g, s, t) for s, t in pairs}
            pairs = [
                p for p in sorted(pair_to_dist, key=pair_to_dist.get, reverse=False)
            ]
        elif self.heuristic == "sound":
            # Idea is to start from backwards.
            # We check for all edges that depend on a clock. clock by clock.
            # We first try eliminating all of them. If it succeeds, then that means
            # that the clock is no longer needed. Else put all the constraints back
            final = [n for n in po.nodes() if len(list(po.successors(n))) == 0][0]
            backward_ordered_nodes = TPOInferenceAlgorithm.get_reachability_order(
                self.graph, final, forward=False
            )
            pairs = [
                (s, t)
                for s in backward_ordered_nodes
                if s in partial_order
                for t in partial_order[s]
            ]
        else:
            raise NotImplementedError("Not Implemented")

        if self.heuristic in ["random", "order", "distant", "near", "sound"]:
            iterator = iter(pairs)

            def next_edge_iter(**kwargs):
                return next(iterator, None)

        return next_edge_iter

    def select_post_processing_func(self):
        """Select a Post Processing Function"""

        if self.heuristic in ["random", "order", "distant", "near"]:
            # No Post Processing Required
            def pass_through_func(*args, **kwargs):
                pass

            func = pass_through_func

        elif self.heuristic == "sound":
            # If all of them are eliminated, the clock is no longer needed.
            #  Else put all the constraints back
            func = PostProcessingFunc(self.partial_order)
        else:
            raise NotImplementedError("Not Implemented")

        return func

    @staticmethod
    def get_event_bounds(
        traces: TimedTraceList, partial_order: Optional[Dict[str, List[str]]] = None
    ) -> NodeBoundDict:
        """Compute min and max time boudn for each event"""
        if partial_order is None:
            inference = POInferenceAlgorithm()
            po = inference.infer(ArrayDataset(traces))
            partial_order: Dict = po.partial_order

        events = set(
            list(partial_order.keys())
            + [t for targets in partial_order.values() for t in targets]
        )
        event_to_times: dict[str, list] = {e: [] for e in events}
        for trace in traces:
            trace_dict: dict[TimeStamp, Symbol] = {s: t for t, s in trace}
            for source_time, source_event in trace:
                event_to_times[source_event].append(source_time)
                if source_event not in partial_order:
                    continue
                for target_event in partial_order[source_event]:
                    if target_event not in trace_dict:
                        continue
                    target_time = trace_dict[target_event]
                    event_to_times[target_event].append(target_time)

        event_to_bound = {
            e: (min(times), max(times)) for e, times in event_to_times.items()
        }
        return event_to_bound

    @staticmethod
    def get_event_pair_bounds(
        traces: TimedTraceList, partial_order: Optional[Dict[str, List[str]]] = None
    ) -> EdgeBoundDict:
        """Compute min and max time boudn for each event"""
        if partial_order is None:
            inference = POInferenceAlgorithm()
            po = inference.infer(traces)
            partial_order = po.partial_order

        events = set(
            list(partial_order.keys())
            + [t for targets in partial_order.values() for t in targets]
        )
        event_pair_to_time_diffs: dict = {}
        for trace in traces:
            trace_dict = {s: t for t, s in trace}
            for source_time, source_event in trace:
                if source_event not in partial_order:
                    continue
                for target_event in partial_order[source_event]:
                    if target_event not in trace_dict:
                        continue
                    target_time = trace_dict[target_event]
                    time_diff = target_time - source_time
                    if (source_event, target_event) not in event_pair_to_time_diffs:
                        event_pair_to_time_diffs[(source_event, target_event)] = []
                    event_pair_to_time_diffs[(source_event, target_event)].append(
                        time_diff
                    )

        event_pair_to_bound = {
            pair: (min(times), max(times))
            for pair, times in event_pair_to_time_diffs.items()
        }
        return event_pair_to_bound

    @staticmethod
    def get_reachability_order(
        graph: nx.DiGraph, init_node: str, forward: bool = True
    ) -> List:
        """Compute a Reachability order of a graph"""
        if forward:
            next_node_attr = "successors"
            visit_node_attr = "predecessors"
        else:
            next_node_attr = "predecessors"
            visit_node_attr = "successors"

        Q = queue.Queue()
        V = set()

        Q.put(init_node)
        V.add(init_node)

        order = [init_node]

        while not Q.empty():
            curr_node = Q.get()
            for next_node in getattr(graph, next_node_attr)(curr_node):
                if next_node not in V and all(
                    [n in V for n in getattr(graph, visit_node_attr)(next_node)]
                ):
                    V.add(next_node)
                    Q.put(next_node)
                    order.append(next_node)
        return order


class TimeConstraintsLP:
    """
    Handle Time Constraints Operation for Linear Programming

    Attributes:
        events                  A set of events
        pairs                   A set of event pairs (source, target)
        event_to_bound          A dictionary that maps an event to (lb, ub)
        event_pair_to_bound     A dictionary that maps an event pair to (lb, ub)

        A, b
        event_to_index
        pair_to_index
    """

    def __init__(
        self,
        traces,
        partial_order,
        decimals: int = 2,
        threshold: float = 0.5,
        slack_threshold: float = 1e-1,
    ):
        self.pairs = [(s, t) for s, ts in partial_order.items() for t in ts]
        self.events = list(set(reduce(lambda x, y: list(x) + list(y), self.pairs)))

        # Get Event -> [LowerBound, UpperBound] and Pair -> [LB, UB]
        self.event_to_bound = TPOInferenceAlgorithm.get_event_bounds(
            traces, partial_order
        )
        self.event_pair_to_bound = TPOInferenceAlgorithm.get_event_pair_bounds(
            traces, partial_order
        )

        self.decimals = decimals
        self.threshold = threshold
        self.slack_threshold = slack_threshold

        num_event = len(self.events)
        num_pair = len(self.pairs)
        num_variable = num_event

        self.A = np.zeros((2 * num_event + 2 * num_pair, num_variable))
        self.b = np.zeros(2 * num_event + 2 * num_pair)

        # Map event to indices e1! -> (0, 1)
        self.event_to_indices = {
            e: (2 * i, 2 * i + 1) for i, e in enumerate(self.events)
        }
        # Map event pair to indices (e1!, e2!) -> (10, 11)
        offset = 2 * num_event
        self.pair_to_indices = {
            (s, t): (offset + 2 * i, offset + 2 * i + 1)
            for i, (s, t) in enumerate(self.pairs)
        }

        self.index_to_event = {
            i: e for e, ii in self.event_to_indices.items() for i in ii
        }
        self.index_to_pair = {
            i: p for p, ii in self.pair_to_indices.items() for i in ii
        }

        # Constrcut the LP constraints when instantiating TimeConstraintSLP
        self.construct_lp_constraints()

    def construct_lp_constraints(self) -> None:
        """Construct Linear Programming Constraints"""

        for event in self.events:
            lb = max(0, self.event_to_bound[event][0])
            ub = self.event_to_bound[event][1]
            self.set_event_bound(event, lb, ub)

        for source_event, target_event in self.pairs:
            lb = max(0, self.event_pair_to_bound[(source_event, target_event)][0])
            ub = self.event_pair_to_bound[(source_event, target_event)][1]
            self.set_pair_bound(source_event, target_event, lb, ub)

    def get_event_row_index(self, event: str, get_lb: bool) -> int:
        """Get an index for the given constraint (row index)"""
        if event not in self.event_to_indices:
            raise ValueError(f"{event} not found in {self.event_to_indices}")

        indices = self.event_to_indices[event]

        if get_lb:
            return indices[0]
        return indices[1]

    def get_pair_row_index(
        self, source_event: str, target_event: str, get_lb: bool
    ) -> int:
        """Get Index of the constraint"""
        pair = (source_event, target_event)
        if pair not in self.pair_to_indices:
            raise ValueError(f"{pair} not found in {self.pair_to_indices}")

        indices = self.pair_to_indices[pair]

        if get_lb:
            return indices[0]
        return indices[1]

    def set_event_bound(
        self, event: str, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> None:
        """Set Event Lower/Upper Bound"""
        col = self.get_column_index(event)
        lb_row = self.get_event_row_index(event, get_lb=True)
        ub_row = self.get_event_row_index(event, get_lb=False)

        if lb is not None:
            self.A[lb_row, col] = -1
            self.b[lb_row] = -lb
        if ub is not None:
            self.A[ub_row, col] = 1
            self.b[ub_row] = ub

    def set_pair_bound(
        self,
        source_event: str,
        target_event: str,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ) -> None:
        """Set Pair Lower/Upper Bound"""
        source_col = self.get_column_index(source_event)
        target_col = self.get_column_index(target_event)
        lb_row = self.get_pair_row_index(source_event, target_event, get_lb=True)
        ub_row = self.get_pair_row_index(source_event, target_event, get_lb=False)

        if lb is not None:
            self.A[lb_row, source_col] = 1
            self.A[lb_row, target_col] = -1
            self.b[lb_row] = -lb
        if ub is not None:
            self.A[ub_row, source_col] = -1
            self.A[ub_row, target_col] = 1
            self.b[ub_row] = ub

    def get_pair_bound(
        self, source_event: str, target_event: str, get_lb: bool
    ) -> float:
        """Get Lower/Upper bound for the event pair"""
        index = self.get_pair_row_index(source_event, target_event, get_lb)
        bound = self.b[index]

        if get_lb:
            return -bound
        return bound

    def get_row(self, source_event: str, target_event: str, get_lb: bool) -> np.ndarray:
        """Get a row of the provided constraint"""
        index = self.get_pair_row_index(source_event, target_event, get_lb)
        return self.A[index, :]

    def get_events_string(self, source_event: str, target_event: str) -> str:
        """Translate a constraint to an strin"""
        row = self.get_row(source_event, target_event, False)

        events = self.get_events(row)
        signs = self.get_signs(row, False)
        signs = ["+" if s > 0 else "-" for s in signs]
        return "".join([f"{s}{v}" for v, s in zip(events, signs)])

    def get_constraint_string(
        self, source_event: str, target_event: str, get_lb: bool, decimals: int
    ) -> str:
        """Get Constraint String"""
        events_str = self.get_events_string(source_event, target_event)
        bound = self.get_pair_bound(source_event, target_event, get_lb)
        bound = np.around(bound, decimals)

        if get_lb:
            return f"{bound} <= {events_str}"

        return f"{events_str} <= {bound}"

    def get_column_index(self, event: str) -> int:
        """Get the event variable index (column index)"""
        return self.events.index(event)

    def get_events(self, row: np.ndarray) -> List:
        """Find which event the indices are refering to"""
        if len(row) != len(self.events):
            raise Exception("The of the indices must be equal to that of No. of events")
        nonzero_indices = np.nonzero(row)[0]
        events = [self.events[i] for i in nonzero_indices]
        return events

    def get_signs(self, row: np.ndarray, get_lb: bool) -> np.ndarray:
        """Get Sign"""
        if len(row) != len(self.events):
            raise Exception("The of the indices must be equal to that of No. of events")
        nonzero_indices = np.nonzero(row)[0]
        if get_lb:
            return -row[nonzero_indices]
        return row[nonzero_indices]

    def get_event_from_row_index(self, index: int) -> str:
        """Get Event from index"""
        if index >= self.A.shape[0]:
            raise Exception(f"index must be less than {self.A.shape[0]}")
        if index not in self.index_to_event:
            raise Exception(f"{index} not in {list(self.index_to_event.keys())}")
        return self.index_to_event[index]

    def get_pair_from_row_index(self, index: int) -> Tuple[str, str]:
        """Get Pair From index"""
        if index >= self.A.shape[0]:
            raise Exception(f"index must be less than {self.A.shape[0]}")
        if index not in self.index_to_pair:
            raise Exception(f"{index} not in {list(self.index_to_pair.keys())}")

        return self.index_to_pair[index]

    def remove_constraint(self, source_event: str, target_event, lb: bool) -> None:
        """Remove the constraint"""
        row_index = self.get_pair_row_index(source_event, target_event, lb)
        self.A[row_index, :] = np.zeros(len(self.events))
        self.b[row_index] = 0

    def get_constraints_without(
        self, source_event: str, target_event: str, get_lb: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get Constraints A and b without the specified constraint"""
        row_index = self.get_pair_row_index(source_event, target_event, get_lb)

        if get_lb:
            A = copy.deepcopy(self.A)
            b = copy.deepcopy(self.b)
            A[row_index, :] = np.zeros(len(self.events))
            b[row_index] = 0  # Replace a lb constraint (>=lb) with >= 0
        else:
            A = copy.deepcopy(self.A)
            b = copy.deepcopy(self.b)
            # Instead of deleting the row (mess up the index), Set the row to zeros
            A[row_index, :] = np.zeros(len(self.events))
            b[row_index] = 0
        return A, b

    def is_redundant(
        self, orig_bound: float, new_bound: float, lb: bool, threshold: float
    ) -> bool:
        """
        Check whther the constraint was redudant by analyzing the changes
        between the original bound and the new bound
        """
        redundant = False
        if new_bound is None:
            return redundant

        # For lb, if the values do not minimize, then it's redundant
        if lb and new_bound > orig_bound - threshold:
            redundant = True
        # For ub, if the values do not maximize
        elif not lb and new_bound < orig_bound + threshold:
            redundant = True
        return redundant

    def solvefor(
        self,
        source_event: str,
        target_event: str,
        lb: bool,
        decimals: Optional[int] = None,
        threshold: Optional[float] = None,
        slack_threshold: Optional[float] = None,
        debug: bool = False,
    ) -> Dict:
        """
        Solve for the specified constraint.
        1. Remove the specified constraint from the constraint list
        2. Optimize for the specified constraint (minimize if lb or else maximize if ub)
        3. If the constraint was identified "redundant", return all the information
        """

        if decimals is None:
            decimals = self.decimals

        if threshold is None:
            threshold = self.threshold

        if slack_threshold is None:
            slack_threshold = self.slack_threshold

        constraint_str = self.get_constraint_string(
            source_event, target_event, lb, decimals
        )

        # index = self.get_pair_row_index(source_event, target_event, lb)
        # row = self.get_row(source_event, target_event, lb)
        # bound = self.get_pair_bound(source_event, target_event, lb)

        A, b = self.get_constraints_without(source_event, target_event, lb)
        c = self.get_row(source_event, target_event, lb)

        sol = cvx.solvers.conelp(
            cvx.matrix(-c),
            cvx.matrix(A),
            cvx.matrix(b),
            options={"show_progress": debug},
        )

        orig_bound = self.get_pair_bound(source_event, target_event, lb)
        if sol is None or sol["primal objective"] is None:
            new_bound = None
            redundant = False
        else:
            new_bound = sol["primal objective"] if lb else -sol["primal objective"]
            redundant = self.is_redundant(orig_bound, new_bound, lb, threshold)

        if redundant:
            if debug:
                print("Redundant Constraint: " + constraint_str)

            graph = nx.DiGraph()
            graph.add_edge(source_event, target_event)
            constraints = defaultdict(lambda: [])
            constraints[(source_event, target_event)].append(lb)

            for index, slack in enumerate(sol["s"]):
                if slack < slack_threshold and len(np.nonzero(A[index, :])[0]) != 0:
                    lb_ = index % 2 == 0
                    if index >= 2 * len(self.events):
                        source, target = self.get_pair_from_row_index(index)
                        graph.add_edge(source, target)
                        constraints[(source, target)].append(lb_)

            nodes = list(nx.shortest_path(graph.to_undirected(), target_event).keys())
            pairs = graph.subgraph(nodes).edges()
            for source, target in pairs:
                for lb_ in set(constraints[(source, target)]):
                    if source == source_event and target == target_event and lb_ == lb:
                        continue
                    constraint_str = self.get_constraint_string(
                        source, target, lb_, decimals
                    )
                    if debug:
                        print(constraint_str)

        return {
            "redundant": redundant,
            "orig_bound": orig_bound,
            "new_bound": new_bound,
        }
