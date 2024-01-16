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
import os
import queue
import random
import re
import subprocess as sp
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import graphviz
import networkx as nx
from IPython.display import Image, display

from specless.dataset import Dataset, PathToFileDataset
from specless.specification import PartialOrder, Specification, TimedPartialOrder
from specless.timeconstraintslp import TimeConstraintsLP
from specless.typing import EdgeBoundDict, NodeBoundDict, TimedTraceList


class InferenceAlgorithm(metaclass=ABCMeta):
    """Base class for the inference algorithms.
    The algorithm infers a specification from demonstrations (dataset).
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args: tuple = args
        self.kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def infer(self, dataset: Dataset) -> Union[Specification, Exception]:
        raise NotImplementedError()


class POInferenceAlgorithm(InferenceAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_partial_order(traces: List[List[str]]) -> Dict[str, List[str]]:
        forwards: defaultdict[Any, list[Any]] = defaultdict(lambda: [])
        backwards: defaultdict[Any, list[Any]] = defaultdict(lambda: [])

        # find forwards and negative backwards
        for trace in traces:
            visited: list = []
            for i, symbol in enumerate(trace):
                if i != 0:
                    for v in visited:
                        if symbol not in forwards[v]:
                            forwards[v].append(symbol)
                        if v not in backwards[symbol]:
                            backwards[symbol].append(v)
                visited.append(symbol)

        return {
            symbol: [s for s in forwards[symbol] if s not in backwards[symbol]]
            for symbol in forwards.keys()
        }

    def infer(self, dataset: Dataset) -> Union[Specification, Exception]:
        traces: List = dataset.tolist(key="symbol")
        partial_order = POInferenceAlgorithm.get_partial_order(traces)

        # add edges
        edges = []
        for symbol, next_symbols in partial_order.items():
            for next_symbol in next_symbols:
                edges.append((symbol, next_symbol))

        po = PartialOrder(edges)
        # It could
        # 1. include redundant edges
        po = nx.transitive_reduction(po)
        # 2. be consisted of multiple independent graphs

        # Finally, return the partial order(s)
        return po


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

    def infer(self, dataset: Dataset) -> Union[Specification, Exception]:
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

        sorted_dataset = dataset.apply(
            lambda data: data.sort_values(by="timestamp", inplace=False)
        )
        traces: List = sorted_dataset.tolist(key="symbol")

        # Find a partial order
        inference = POInferenceAlgorithm()
        partial_order: Dict = inference.get_partial_order(traces)
        po: PartialOrder = inference.infer(sorted_dataset)
        # Infer Timing Constraints
        global_constraints, local_constraints = self.infer_time_constraints(
            traces, po, partial_order
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
        lp: TimeConstraintsLP,
        po: PartialOrder,
        partial_order: Dict[str, list[str]],
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
            po = inference.infer(traces)
            partial_order = po.partial_order

        events = set(
            list(partial_order.keys())
            + [t for targets in partial_order.values() for t in targets]
        )
        event_to_times: dict[str, list] = {e: [] for e in events}
        for trace in traces:
            trace_dict = {s: t for t, s in trace}
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


class AutomataInferenceAlgorithm(InferenceAlgorithm):
    """The inference algorithm for inferring an automaton from a list of Traces,
    where trace is defined as a sequence of symbols, i.e. a set of strings.
    For example, ${a, b, c}$

    Args:
        InferenceAlgorithm (_type_): _description_
    """

    def __init__(
        self, binary_location: str = "dfasat/flexfringe", output_directory: str = "./"
    ) -> None:
        """FlexFringe Interface. Directly access the binary via bash commands

        Args:
            binary_location (str, optional): (absolute / relative) filepath to the
                                       flexfringe binary. Defaults to "dfasat/flexfringe".
            output_directory (str, optional): The flexfringe output directory. Defaults to "./".
        """
        super().__init__()
        self.binary_location = binary_location
        self.num_training_examples: int
        self.num_symbols: int
        self.total_symbols_in_examples: int
        self.output_filepath: str
        self.learned_model_filepath: str
        self.initial_model_filepath: str
        self.average_elapsed_time: float = 0

        self._output_filename: str = "dfa"
        self._final_output_addon_name: str = "final"
        self._learned_model_filepath: str
        self._initial_output_addon_name = "init_dfa"
        self._initial_model_filepath: str
        self._output_base_filepath: Optional[str] = None
        self._output_directory: str = output_directory
        self._flexfringe_output_dir_popt_str: str = "o"
        Path(output_directory).mkdir(parents=True, exist_ok=True)

    def infer(
        self,
        dataset: Dataset,
        get_help: bool = False,
        record_time: bool = True,
        go_fast: bool = False,
        **kwargs,
    ) -> Union[Specification, Exception]:
        """calls the flexfringe binary given the data in the training file

        Args:
            dataset (Dataset): dataset that contains the path to the training data
            get_help (bool, optional): Whether or not to print the flexfringe
                                    usage help memu. Defaults to False.
            record_time (bool, optional): _description_. Defaults to True.
            go_fast (bool, optional): optimizes this call to make it as fast
                                    as possible, at the expensive of usability.
                                    use for benchmarking / Hyperparam
                                    optimization. Defaults to False.
            kwargs (dict, optional) keyword arguments to pass to flexfringe
                                    controlling the learning process

        Raises:
            Exception: _description_

        Returns:
            Union[Specification, Exception]:
        """
        error_msg = "must be an instance of the PathToFileDataset class"
        assert isinstance(dataset, PathToFileDataset), error_msg
        training_file: str = dataset.filepath

        cmd: list = self._get_command(kwargs)
        output_file: str = self.learned_model_filepath

        if get_help:
            flexfringe_call = [self.binary_location] + cmd + [""]
        else:
            flexfringe_call = [self.binary_location] + cmd + [training_file]

            if not go_fast:
                # get summary statistics of learning data and save them for
                # later use of the inference interface
                with open(training_file) as fh:
                    content = fh.readlines()
                    first_line = content[0]
                    N, num_symbols_str = re.match(r"(\d*)\s(\d*)", first_line).groups()
                    self.num_training_examples = int(N)
                    self.num_symbols = int(num_symbols_str)

                    self.total_symbols_in_examples = 0
                    if self.num_training_examples > 0:
                        for line in content[1:]:
                            _, line_len, _ = re.match(
                                r"(\d)\s(\d*)\s(.*)", line
                            ).groups()
                            self.total_symbols_in_examples += int(line_len)

        if output_file is not None:
            try:
                os.remove(output_file)
            except OSError:
                pass

        if go_fast:
            stdout = sp.DEVNULL
        else:
            stdout = sp.PIPE

        if record_time:
            start_time = time.time()

        completed_process = sp.run(flexfringe_call, stdout=stdout, stderr=sp.PIPE)

        if not go_fast:
            call_string = completed_process.stdout.decode()
            print("%s" % call_string)

        if record_time:
            elapsed_time = time.time() - start_time
            n_run = int(kwargs["n"]) if "n" in kwargs else 1
            self.average_elapsed_time = elapsed_time / n_run

        if not go_fast:
            model_data = self._read_model_data(output_file)
            if model_data is not None:
                # return model_data
                return Specification()
        raise Exception("No model output generated")

    def draw_IPython(self, filename: str) -> None:
        """Draws the dot file data in a way compatible with a jupyter / IPython
        notebook

        Args:
            filename (str): The learned model dot file data
        """
        dot_file_data = self._read_model_data(filename)

        if dot_file_data == "":
            pass
        else:
            filename = Path(filename).stem
            output_file = os.path.join(self._output_directory, filename)
            g = graphviz.Source(dot_file_data, filename=output_file, format="png")
            g.render()
            display(Image(g.render()))

    def draw_initial_model(self) -> None:
        """
        Draws the initial (prefix-tree) model
        """

        dot_file = self.initial_model_filepath
        self.draw_IPython(dot_file)

    def draw_learned_model(self) -> None:
        """
        Draws the final, learned model
        """

        dot_file = self.learned_model_filepath
        self.draw_IPython(dot_file)

    @property
    def output_filepath(self) -> str:
        """The output filepath for the results of learning the model"""
        self._output_base_filepath = os.path.join(
            self._output_directory, self._output_filename
        )

        return self._output_base_filepath

    @output_filepath.setter
    def output_filepath(self, filepath: str) -> None:
        """sets output_filepath and output_directory based on the given filepath

        Args:
            filepath (str): The new filepath
        """
        (self._output_directory, self._output_base_filepath) = os.path.split(filepath)

    @property
    def learned_model_filepath(self) -> str:
        """the output filename for the fully learned model, as this is a
        different from the inputted "output-dir"

        Returns:
            str: The learned model filepath.
        """
        addon_name = self._final_output_addon_name
        self._learned_model_filepath = self._get_model_file(addon_name)

        return self._learned_model_filepath

    @learned_model_filepath.setter
    def learned_model_filepath(self, filepath: str) -> None:
        """sets the learned_model_filepath and the base model's filepath

        Args:
            filepath (str): The new learned model filepath.
        """
        addon_name = self._final_output_addon_name
        base_model_filepath = self._strip_model_file(filepath, addon_name)

        self._learned_model_filepath = filepath
        self.output_filepath = base_model_filepath

    @property
    def initial_model_filepath(self) -> str:
        """the output filename for the unlearned, initial model, as this is a
        different from the inputted "output-dir".
        In this case, it will be a prefix tree from the given learning data.

        Returns:
            str: The initial model filepath.
        """
        addon_name = self._initial_output_addon_name
        self._initial_model_filepath = self._get_model_file(addon_name)

        return self._initial_model_filepath

    @initial_model_filepath.setter
    def initial_model_filepath(self, filepath: str) -> None:
        """sets the initial_model_filepath and the base model's filepath

        Args:
            filepath (str): The new initial model filepath.
        """
        addon_name = self._initial_model_filepath
        base_model_filepath = self._strip_model_file(filepath, addon_name)

        self._learned_model_filepath = filepath
        self.output_filepath = base_model_filepath

    def _get_model_file(self, addon_name: str) -> str:
        """Gets the full model filepath, with the model type given by addon_name.

        Args:
            addon_name (str): The name to append to the base model name
                                 to access the certain model file

        Returns:
            str: The full model filepath string.
        """
        filepath = self.output_filepath
        f_dir, _ = os.path.split(filepath)

        full_model_filename = self._get_model_filename(addon_name)
        full_model_filepath = os.path.join(f_dir, full_model_filename)

        return full_model_filepath

    def _strip_model_file(self, model_filepath: str, addon_name: str) -> str:
        """Strips the full model filepath of its addon_name to get the base model
        filepath

        Args:
            model_filepath (str): The full model filepath
            addon_name (str): :      The name to strip from the full model file

        Returns:
            str: The base model filepath string.
        """
        f_dir, full_fname = os.path.split(model_filepath)
        fname, ext = os.path.splitext(full_fname)

        # base filepath is just the basename, before the "addon" model type
        # is added to the base model name
        if fname.endswith(addon_name):
            fname = fname[: -len(addon_name)]

        base_model_filepath = os.path.join(f_dir, fname)

        return base_model_filepath

    def _get_model_filename(self, addon_name: str) -> str:
        """Gets the model filename, with the model type given by addon_name.

        Args:
            addon_name (str): The name to append to the base model name
                                 to access the certain model file

        Returns:
            str: The model filename string.
        """
        filepath = self.output_filepath
        f_dir, full_fname = os.path.split(filepath)
        fname, ext = os.path.splitext(full_fname)

        full_model_filename = fname + addon_name + ".dot"

        return full_model_filename

    def _read_model_data(self, model_file: str) -> Union[str, Exception]:
        """Reads in the model data as a string.

        Args:
            model_file (str): The model filepath

        Raises:
            Exception: _description_

        Returns:
            Union[str, Exception]: The model data as a string
        """

        try:
            with open(model_file) as fh:
                return fh.read()

        except FileNotFoundError:
            raise Exception("No model file was found.")

    def _get_command(self, kwargs: dict) -> list:
        """Gets a list of popt commands to send the binary

        Args:
            kwargs (dict): The flexfringe tool keyword arguments

        Returns:
            list: The list of commands.
        """

        # default argument is to print the program's man page
        if len(kwargs) > 1:
            cmd = ["-" + key + "=" + kwargs[key] for key in kwargs]

            # need to give the output directory only if the user hasn't already
            # put that in kwargs.
            if self._flexfringe_output_dir_popt_str not in kwargs:
                cmd += ["--output-dir={}".format(self.output_filepath)]
            else:
                key = self._flexfringe_output_dir_popt_str
                self.output_filepath = kwargs[key]
        else:
            cmd = ["--help"]
            print("no learning options specified, printing tool help:")

        return cmd


class PostProcessingFunc:
    """Post Processing Function Class"""

    def __init__(self, partial_order):
        self.curr_node = None
        self.target_nodes = []
        self.redundants = []
        self.edge_to_bound = {}
        self.partial_order = partial_order

    def __call__(
        self, lp, source_event, target_event, redundant, lb, ub, *args, **kwargs
    ):
        """Post processing function call"""
        if source_event != self.curr_node:
            self.curr_node = source_event
            self.target_nodes = []
            self.redundants = []
            self.edge_to_bound = {}
        self.target_nodes.append(target_event)
        self.redundants.append(redundant)
        self.edge_to_bound[(source_event, target_event)] = (lb, ub)
        if set(self.target_nodes) == set(self.partial_order[source_event]):
            # If all of the constraints are redundant, keep it as it is
            # (kept the constraints removed)
            if sum(self.redundants) == len(self.partial_order[source_event]):
                pass
            # Else put all the contraints back
            else:
                for target_node in self.target_nodes:
                    lb, ub = self.edge_to_bound[(source_event, target_node)]
                    lp.set_pair_bound(source_event, target_node, lb, ub)
