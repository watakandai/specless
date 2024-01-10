# 3rd-party packages
import collections
import copy
import multiprocessing
import os
import queue
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Tuple

import graphviz as gv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from bidict import bidict
from IPython.display import Image, display
from joblib import Parallel, delayed
from networkx.drawing.nx_pydot import to_pydot
from numpy.random import RandomState
from pydot import Dot
from scipy.stats import rv_discrete

from .mps import (
    SWDFA_MPS,
    BMPS_exact,
    MPSReturnData,
    postprocess_MPS,
    should_use_BMPS_exact,
)

# local packages / modules
from .types import (
    GeneratedTraceData,
    Node,
    Nodes,
    NXEdgeList,
    NXNodeList,
    Observation,
    Probabilities,
    Probability,
    SampledTransData,
    Symbol,
    Symbols,
    Trans_data,
)

# needed for multi-threaded sampling routine
NUM_CORES = multiprocessing.cpu_count()

# constants
SMOOTHING_AMOUNT = 0.0001
DEFAULT_FINAL_TRANS_SYMBOL = "$"
DEFAULT_EMPTY_TRANS_SYMBOL = "lambda"


class Automaton(nx.MultiDiGraph, metaclass=ABCMeta):
    """
    This class describes a automaton with (possibly) stochastic transitions

    built on networkx, so inherits node and edge data structure definitions

    Node Attributes
    -----------------
        - final_probability: final state probability for the node
        - trans_distribution: a sampled-able function to select the next state
          and emitted symbol
        - is_accepting: a boolean flag determining whether the automaton
          considers the node accepting

    Edge Properties
    -----------------
        - symbol: the symbol value emitted when the edge is traversed
        - probability: the probability of selecting this edge for traversal

    :param      nodes:                     node list as expected by
                                           networkx.add_nodes_from()
    :param      edge_list:                 edge list as expected by
                                           networkx.add_edges_from()
    :param      symbol_display_map:        bidirectional mapping of
                                           hashable symbols, to a unique
                                           integer index in the symbol map.
                                           Needed to translate between the
                                           indices in the transition
                                           distribution and the hashable
                                           representation which is
                                           meaningful to the user
    :param      alphabet_size:             number of symbols in automaton
    :param      num_states:                number of states in automaton
                                           state space
    :param      start_state:               unique start state string label
                                           of automaton
    :param      smooth_transitions:        whether to smooth the symbol
                                           transitions distributions
    :param      is_stochastic:             the transitions are
                                           non-probabilistic, so we are
                                           going to assign a uniform
                                           distribution over all symbols
                                           for the purpose of generation
    :param      is_sampleable:             will formalize / create probability
                                           distributions for each state's
                                           transitions to allow for sampling of
                                           runs from the machine
    :param      is_normalized:             whether the
                                           edge probabilities are renormalize
                                           such that each states has a well-
                                           defined transition
                                           probability distribution.
                                           We typically DONT want to
                                           modify the probabilities,
                                           except if we would like
                                           to be able to sample traces
    :param      num_obs:                   number of observation symbols
    :param      final_transition_sym:      representation of the
                                           termination symbol
    :param      empty_transition_sym:      representation of the empty
                                           symbol (a.k.a. lambda)
    :param      initial_weight_key:        key in the automaton's node data
                                           corresponding to the weight /
                                           probability of starting in that
                                           node. If None, don't include
                                           this info in the display of the
                                           automaton.
    :param      final_weight_key:          key in the automaton's node data
                                           corresponding to the weight /
                                           probability of ending in that
                                           node. If None, don't include
                                           this info in the display of the
                                           automaton.
    :param      state_observation_key:     The key in each node's data dict
                                           for state observations. If None,
                                           don't include this info in the
                                           display of the automaton
    :param      can_have_accepting_nodes:  Indicates if the automata can
                                           have accepting nodes
    :param      merge_sinks:               whether to combine all states
                                           together that have no outgoing edges
    :param      edge_weight_key:           The key in each edge's data dict
                                           for edge weight / prob. If None,
                                           don't include this info in the
                                           display of the automaton
    :param      smoothing_amount:          probability mass to re-assign to
                                           unseen symbols at each node
    """

    # file I/O
    automata_display_data_dir_name = "automata_images"
    automata_data_dir = "automaton_data"

    def __init__(
        self,
        nodes: NXNodeList,
        edge_list: NXEdgeList,
        symbol_display_map: bidict,
        alphabet_size: int,
        num_states: int,
        start_state: Hashable,
        smooth_transitions: bool,
        is_stochastic: bool,
        is_sampleable: bool,
        is_normalized: bool = False,
        num_obs: {int, None} = None,
        final_transition_sym: Hashable = DEFAULT_FINAL_TRANS_SYMBOL,
        empty_transition_sym: Hashable = DEFAULT_EMPTY_TRANS_SYMBOL,
        initial_weight_key: str = None,
        final_weight_key: str = None,
        state_observation_key: str = None,
        can_have_accepting_nodes: bool = True,
        merge_sinks: bool = False,
        edge_weight_key: str = None,
        smoothing_amount: float = SMOOTHING_AMOUNT,
    ) -> None:
        self._transition_map = dict()
        """a map of start state label and symbol to destination state"""

        self._edge_key_map = dict()
        """mapping between all outgoing transitions and the
        networkx adjacency dictionary keys."""

        self._symbol_display_map = symbol_display_map
        """bidirectional mapping from symbol labels to an int index in
           transition dists."""

        self.alphabet_size = alphabet_size
        """number of symbols in automaton alphabet"""

        self.num_states = num_states
        """number of states in automaton state space"""

        self.num_obs = num_obs
        """number of state observations in TS obs. space"""

        self.final_transition_sym = final_transition_sym
        """representation of the termination symbol"""

        self.empty_transition_sym = empty_transition_sym
        """symbol to use as the empty (a.k.a. lambda) symbol"""

        self.start_state = start_state
        """unique start state string label of automaton"""

        self.is_stochastic = is_stochastic
        """whether symbol probabilities are given for string generation"""

        self._use_smoothing = smooth_transitions
        """whether or not to smooth the input sym. transition distributions"""

        self._smoothing_amount = smoothing_amount
        """probability mass to re-assign to unseen symbols at each node"""

        self.is_sampleable = is_sampleable
        """transitions will have pre-computed, well-formed distributions"""

        self.is_normalized = is_normalized
        """ill-defined transition distributions are normalized to be proper
           probability distributions over outgoing transitions"""

        self.symbols = set()
        """set of all symbols used by the automaton"""

        self.state_labels = set()
        """set of all states in the automaton"""

        self.observations = set()
        """the set of all possible state output symbols (observations)"""

        self.is_deterministic = True
        """whether or not there is a unique state dest. under each symbol.
           Defaults to true and is falsified later during initialization."""

        self._transition_matrices = dict()
        """a dict (keyed on symbol) of (num_states x num_states)
           probabilistic transition matrix
           (NOT always a proper stochastic mat)"""

        self._node_index_map = bidict()
        """a mapping from node label to it's index in the vectorized
           representation of the automaton"""

        self._initial_state_distribution: np.ndarray
        """a (1 x num_states) ndarray containing the pmf for the initial
           starting state. For most machines, this simply the indicator
           function with a one at the index of the state"""

        self._final_state_distribution: np.ndarray
        """a (num_states x 1) ndarray containing the pmf for terminating
           at each state's index."""

        self._automata_display_dir = os.path.join(
            self.automata_data_dir, self.automata_display_data_dir_name
        )
        Path(self._automata_display_dir).mkdir(parents=True, exist_ok=True)
        """the base directory for all output data for the automaton"""

        # need to start with a fully initialized networkx digraph
        super().__init__()

        self.add_nodes_from(nodes)
        self.add_edges_from(edge_list)

        self._initialize_node_edge_properties(
            state_observation_key=state_observation_key,
            final_weight_key=final_weight_key,
            initial_weight_key=initial_weight_key,
            can_have_accepting_nodes=can_have_accepting_nodes,
            edge_weight_key=edge_weight_key,
            merge_sinks=merge_sinks,
        )

    def disp_edges(self, graph: {None, nx.MultiDiGraph} = None) -> None:
        """
        Prints each edge in the graph in an edge-list tuple format

        :param      graph:  The graph to access. Default = None => use instance
        :type       graph:  {None, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        for node, neighbors in graph.adj.items():
            for neighbor, edges in neighbors.items():
                for edge_number, edge_data in edges.items():
                    print(node, neighbor, edge_data)

    def disp_nodes(self, graph: {None, nx.MultiDiGraph} = None) -> None:
        """
        Prints each node's data view

        :param      graph:  The graph to access. Default = None => use instance
        :type       graph:  {None, nx.MultiDiGraph}
        """

        if graph is None:
            graph = self

        for node in graph.nodes(data=True):
            print(node)

    def draw(
        self,
        filename: {str, None} = None,
        should_display: bool = True,
        img_format="png",
    ) -> None:
        """
        Draws (can save) the automaton structure in a way compatible with a
        jupyter / IPython notebook

        :param      filename:  The filename to save the automaton image
        """

        graph = self._get_pydot_representation()

        if filename:
            graph = gv.Source(graph)
            fpath = os.path.join(self._automata_display_dir, filename)
            path = graph.render(format=img_format, filename=fpath)

            if should_display:
                display(Image(filename=path))
        else:
            dot_string = graph.to_string()

            if should_display:
                display(gv.Source(dot_string))

    def plot_node_trans_dist(self, curr_state: Node) -> None:
        """
        Plots the transition pmf at the given curr_state / node.

        :param      curr_state:  state to display its transition distribution
        :type       curr_state:  Hashable
        """

        trans_dist = self._get_node_data(curr_state, "trans_distribution")
        symbols = self._convert_symbol_idxs(trans_dist.xk)

        fig, ax = plt.subplots(1, 1)
        ax.plot(symbols, trans_dist.pmf(trans_dist.xk), "ro", ms=12, mec="r")
        ax.vlines(symbols, 0, trans_dist.pmf(trans_dist.xk), colors="r", lw=4)
        plt.show()

    @classmethod
    def write_traces_to_file(
        cls,
        traces: List[Symbols],
        file: str,
        alphabet_size: int,
        base_file_dir: {str, None} = None,
    ) -> str:
        """
        Writes trace samples to a file in the abbadingo format for use in
        grammatical inference tools like flexfringe

        :param      traces:         The traces to write to a file
        :param      file:           The file name to write to. Can be a partial
                                    path.
        :param      alphabet_size:  The alphabet size
        :param      base_file_dir:  Provide this if you want to output the file
                                    to a different location than
                                    self.automata_data_dir.

        :returns:   the absolute filepath to the traces. Will be:
                    abs_filepath(self.automata_data_dir/file) if base_file_dir
                    is None. Else, will be: abs_filepath(base_file_dir/file)
        """

        # make sure the output traces always go to the automaton's data dir
        if base_file_dir is None:
            base_file_dir = cls.automata_data_dir

        filepath = os.path.join(base_file_dir, file)
        file_dir, _ = os.path.split(filepath)
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        # make sure the num_samples is an int, so you don't have to wrap shit
        # in an 'int()' every time...
        num_samples = len(traces)

        with open(filepath, "w+") as f:
            # need the header to be:
            # number_of_training_samples size_of_alphabet
            f.write(str(num_samples) + " " + str(alphabet_size) + "\n")

            for trace in traces:
                trace_length = len(trace)
                f.write(
                    cls._get_abbadingo_string(trace, trace_length, is_pos_example=True)
                )

        return os.path.abspath(filepath)

    def generate_traces(
        self,
        num_samples: int,
        N: int,
        max_resamples: int = 10,
        return_whatever_you_got: bool = False,
        force_multicore: bool = False,
        verbose: int = 50,
    ) -> GeneratedTraceData:
        """
        generates num_samples random traces from the automaton

        :param      num_samples:              The number of trace samples to
                                              generate
        :param      N:                        maximum length of trace
        :param      max_resamples:            The maximum number of times to
                                              resample if if we create a trace
                                              of length N that still doesn't
                                              have a probability > 0 in the
                                              language
        :param      return_whatever_you_got:  Whether to return a string with a
                                              zero probability after all
                                              resampling attempts are
                                              exhausted.
        :param      force_multicore:          whether to force use the threaded
                                              sampler this is set by default to
                                              optimize speed, as the threaded
                                              sampler is slower for smaller
                                              num_samples. Force this to be
                                              true if the automaton is slow to
                                              sample.
        :param      verbose:                  verbose for joblib.Parallel

        :returns:   list of sampled traces, list of the associated trace
                    lengths, list of the associated trace probabilities
        :rtype:     tuple(list(list(int)), list(int), list(float))
        """

        start_state = self.start_state

        # make sure the num_samples is an int, so you don't have to wrap shit
        # in an 'int()' every time...
        num_samples = int(num_samples)

        # dont start a parallel job unless it's a very large one
        iters = range(0, num_samples)
        if (num_samples < 500 or NUM_CORES == 1) and not force_multicore:
            results = [
                self.generate_trace(
                    start_state, N, max_resamples, return_whatever_you_got
                )
                for i in iters
            ]
        else:
            runner = Parallel(n_jobs=NUM_CORES, verbose=verbose)
            job = delayed(self.generate_trace)
            results = runner(
                job(start_state, N, max_resamples, return_whatever_you_got)
                for i in iters
            )

        no_strings_found = all(result == (None, None, None) for result in results)
        if not no_strings_found:
            # remove any None items resulting from failed resampling
            num_failed_samples = 0
            samples, trace_lengths, trace_probs = [], [], []

            for sample, trace_length, trace_prob in results:
                if sample is not None:
                    samples.append(sample)
                    trace_lengths.append(trace_length)
                    trace_probs.append(trace_prob)
                else:
                    num_failed_samples += 1

            if num_failed_samples > 0:
                num_good_samples = num_samples - num_failed_samples
                msg = (
                    f"only sampled {num_good_samples} non-zero "
                    + f"probability strings when {num_samples} strings "
                    + "were requested. Try increasing "
                    + f"the number of resample attempts from {max_resamples}."
                )
                warnings.warn(msg)
        else:
            msg = (
                "No non-zero probability strings found. Try increasing "
                + f"the number of resample attempts from {max_resamples}."
            )
            warnings.warn(msg)

            samples, trace_lengths, trace_probs = results[0]

        return samples, trace_lengths, trace_probs

    def generate_trace(
        self,
        start_state: Node,
        N: int,
        max_resamples: int = 10,
        return_whatever_you_got: bool = False,
        random_state: {None, int, Iterable} = None,
    ) -> (Symbols, int, Probability):
        """
        Generates a trace w/ prob. > 0 from the automaton from its start_state

        :param      start_state:              the state label to start sampling
                                              traces from
        :param      N:                        maximum length of trace
        :param      max_resamples:            The maximum number of times to
                                              resample if if we create a trace
                                              of length N that still doesn't
                                              have a probability > 0 in the
                                              language
        :param      return_whatever_you_got:  Whether to return a string with a
                                              zero probability after all
                                              resampling attempts are
                                              exhausted.
        :param      random_state:             The np.random.RandomState() seed
                                              parameter for sampling from the
                                              state transition distribution.
                                              Defaulting to None causes the
                                              seed to reset.

        :returns:   the sequence of symbols emitted, the length of the trace,
                    the probability of the trace in the language of the
                    automaton

        :raises     ValueError:               if you try to generate a trace
                                              from a non-sampleable automaton
        """

        if not self.is_sampleable:
            msg = "Cannot generate traces in a non-sampleable automaton"
            raise ValueError(msg)

        curr_state = start_state
        length_of_trace = 1
        trace_prob = 1.0
        num_times_restarted = 0

        (next_state, next_symbol, trans_probability) = self._choose_next_state(
            curr_state, random_state
        )

        sampled_trace = [next_symbol]
        curr_state = next_state
        at_terminal_state = next_symbol == self.final_transition_sym
        trace_prob *= trans_probability

        while not at_terminal_state:
            (next_state, next_symbol, trans_probability) = self._choose_next_state(
                curr_state, random_state
            )

            curr_state = next_state
            trace_prob *= trans_probability

            if next_symbol == self.final_transition_sym:
                break

            sampled_trace.append(next_symbol)
            length_of_trace += 1

            # we need to generate a trace with probability > 0, so if we
            # hit the max trace length limit while sampling, we need to try
            # sampling a whole new trace again.
            if length_of_trace == N:
                num_times_restarted += 1

                if num_times_restarted == max_resamples:
                    msg = (
                        "tried resampling a non-zero probability trace "
                        + f"{max_resamples} times and failed. "
                        + "Try increasing the current max trace length "
                        + f"{N} and checking that at least one reachable "
                        + "state has a non-zero final-state probability."
                    )
                    warnings.warn(msg, RuntimeWarning)

                    if return_whatever_you_got:
                        trace_prob *= 0.0
                        break
                    else:
                        return None, None, None
                else:
                    curr_state = start_state
                    length_of_trace = 0
                    trace_prob = 1.0
                    sampled_trace = []

        return sampled_trace, length_of_trace, trace_prob

    def observe(self, curr_state: Node) -> Observation:
        """
        Returns the given state's observation symbol

        :param      curr_state:  The current TS state

        :returns:   observation symbol emitted at curr_state
        """

        raise NotImplementedError

    def add_node(self, node_for_adding, **attr):
        """
        Add a single node `node_for_adding` and update node attributes.

        Parameters
        ----------
        node_for_adding : node
            A node can be any hashable Python object except None.
        attr : keyword arguments, optional
            Set or change node attributes using key=value.

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_node(1)
        >>> G.add_node('Hello')
        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
        >>> G.add_node(K3)
        >>> G.number_of_nodes()
        3

        Use keywords set/change node attributes:

        >>> G.add_node(1, size=10)
        >>> G.add_node(3, weight=0.4, UTM=('13S', 382871, 3972649))

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.

        On many platforms hashable items also include mutables such as
        NetworkX Graphs, though one should be careful that the hash
        doesn't change on mutables.
        """

        self.num_states += 1

        return super(nx.MultiDiGraph, self).add_node(node_for_adding, **attr)

    def most_probable_string(
        self,
        min_string_probability: {Probability, None} = None,
        max_string_length: {int, None} = None,
        allow_empty_symbol: bool = False,
        try_to_use_greedy: bool = True,
        backwards_search: bool = True,
        num_strings_to_find: int = 1,
        depth_first: bool = False,
        add_entropy: bool = False,
        disable_pbar: bool = False,
    ) -> MPSReturnData:
        """
        Computes the bounded, most probable string in the probabilistic
        language of the automaton.

        :param      min_string_probability:  The minimum string probability.
                                             This setting does nothing if
                                             is_deterministic, as the
                                             deterministic algorithm is exact.
                                             (default 0.0)
        :param      max_string_length:       The maximum string length. This
                                             setting does nothing if
                                             is_deterministic, as the
                                             deterministic algorithm is exact.
                                             (default 100)
        :param      allow_empty_symbol:      Indicates if the empty symbol is
                                             allowed
        :param      try_to_use_greedy:       whether to try using the MUCH
                                             faster greedy search algorithm.
                                             only possible if the automaton has
                                             deterministic transitions. Only
                                             set this to False if the automaton
                                             actually is non-deterministic, as
                                             the non-deterministic solver is an
                                             approximation and MUCH slower.
        :param      backwards_search:        Whether to search from the with
                                             final probability back to the
                                             start state. Often will improve
                                             performance.
        :param      num_strings_to_find:     The number of viable strings to
                                             return. Defaults to only return
                                             the ONE, highest probability
                                             string encountered thus far in the
                                             search, which means the algorithm
                                             is the original BMPS_exact. If >1,
                                             then the algorithm returns the
                                             num_strings_to_find most probable,
                                             viable strings from the search
                                             heap.
        :param      depth_first:             Whether to explore the automaton
                                             using a depth-first search
                                             pattern. Using a depth-first
                                             search pattern will be faster for
                                             very deep, tree-shaped automaton,
                                             but will not return the absolute
                                             best symbol sequence for the given
                                             min_string_prob and
                                             max_string_length. Only turn on if
                                             you have a terminal states deep in
                                             the automaton and you need the
                                             search to be faster.
        :param      add_entropy:             Only keeps a new viable string if
                                             it has a previously unseen
                                             probability of being generated
        :param      disable_pbar:            Disable pbar for speeding up the
                                             computation speed.

        :returns:   most probable string, probability of producing the most
                    probable string, num_strings_to_find (their probs., viable
                    strings) ranked by each string's probability.

        :raises     ValueError:              Cannot be computed for
                                             non-stochastic automaton
        """

        if not self.is_stochastic:
            msg = (
                "Cannot compute most probable string for a "
                + "non-stochastic automaton"
            )
            raise ValueError(msg)

        # setting default values in case they're not given
        if min_string_probability is None:
            min_string_probability = 0.0

        if max_string_length is None:
            max_string_length = 100

        use_BMPS_exact = should_use_BMPS_exact(
            num_strings_to_find, try_to_use_greedy, self.is_deterministic
        )

        # if the empty symbol isn't allowed to be in the MPS, then we should
        # remove it from all of the viable symbols before we even start the
        # search
        empty_symbol = self.empty_transition_sym
        if allow_empty_symbol:
            symbols = [symbol for symbol in self.symbols]
        else:
            symbols = [symbol for symbol in self.symbols if symbol != empty_symbol]

        if use_BMPS_exact:
            params = self._get_BMPS_exact_params(
                symbols,
                max_string_length,
                min_string_probability,
                num_strings_to_find,
                backwards_search,
                allow_empty_symbol,
                depth_first,
                add_entropy,
                disable_pbar,
            )
            (mps, prob, viable_symbols) = BMPS_exact(**params)

        else:
            trans_prob_fcn = self._get_trans_probabilities
            (mps, prob, viable_symbols) = SWDFA_MPS(
                states=self.state_labels,
                start_state=self.start_state,
                F=self._final_state_distribution,
                empty_symbol=empty_symbol,
                node_index_map=self._node_index_map,
                trans_prob_fcn=trans_prob_fcn,
                transition_map=self._transition_map,
            )

        return postprocess_MPS(
            mps,
            prob,
            viable_symbols,
            self._convert_symbol_idxs,
            use_BMPS_exact,
            allow_empty_symbol,
            backwards_search,
        )

    def _get_BMPS_exact_params(
        self,
        symbols: Symbols,
        max_string_length: int,
        min_string_probability: Probability,
        num_strings_to_find: int,
        backwards_search: bool,
        allow_empty_symbol: bool,
        depth_first: bool,
        add_entropy: bool,
        disable_pbar: bool,
    ) -> dict:
        """
        Gets the BMPS_exact algorithm's parameters from the current automaton
        and the algorithm's desired usage.

        :param      symbols:                 Candidate symbols for the MPS
        :param      max_string_length:       The maximum string length.
        :param      min_string_probability:  The minimum string probability.
        :param      num_strings_to_find:     The number of viable strings to
                                             return. Defaults to only return
                                             the ONE, highest probability
                                             string encountered thus far in the
                                             search, which means the algorithm
                                             is the original BMPS_exact. If >1,
                                             then the algorithm returns the
                                             num_strings_to_find most probable,
                                             viable strings from the search
                                             heap.
        :param      backwards_search:        Whether to search from the with
                                             final probability back to the
                                             start state. Often will improve
                                             performance.
        :param      allow_empty_symbol:      Indicates if the empty symbol is
                                             allowed
        :param      depth_first:             Whether to explore the automaton
                                             using a depth-first search
                                             pattern. Using a depth-first
                                             search pattern will be faster for
                                             very deep, tree-shaped automaton,
                                             but will not return the absolute
                                             best symbol sequence for the given
                                             min_string_prob and
                                             max_string_length. Only turn on if
                                             you have a terminal states deep in
                                             the automaton and you need the
                                             search to be faster.
        :param      add_entropy:             Only keeps a new viable string if
                                             it has a previously unseen
                                             probability of being generated
        :param      disable_pbar:            Disable pbar for speeding up the
                                             computation speed.

        :returns:   The BMPS_exact parameters dict.
        """

        empty_symbol = self.empty_transition_sym
        empty_sym_idx = self._symbol_display_map[empty_symbol]

        # numba pre-processing
        symbol_idxs = [self._symbol_display_map[symbol] for symbol in symbols]

        trans_mat_dict = self._transition_matrices
        num_states = self.num_states
        trans_mat = np.empty((num_states, num_states, self.alphabet_size))

        min_string_prob = min_string_probability

        for sym_idx in symbol_idxs:
            symbol = self._convert_symbol_idxs(sym_idx)
            trans_mat[:, :, sym_idx] = trans_mat_dict[symbol]

        if backwards_search:
            # we are going to change the search from start to goal, to goal
            # to start search, as we postulate the product is somewhat
            # tree-shaped

            # # this transposes each d x d transition matrix
            # for sym_idx in symbol_idxs:
            #     symbol = self._convert_symbol_idxs(sym_idx)
            #     trans_mat[:, :, sym_idx] = trans_mat_dict[symbol].T
            trans_mat = trans_mat.swapaxes(0, 1)

            S = self._final_state_distribution.T
            F = self._initial_state_distribution.T

        else:
            S = self._initial_state_distribution
            F = self._final_state_distribution

        # the empty string is always the first symbols out of both of these
        # solvers, so if it's not allowed, we need to generate a string that is
        # one longer
        if not allow_empty_symbol:
            max_string_length += 1

        params = {
            "symbols": symbol_idxs,
            "M": trans_mat,
            "S": S,
            "F": F,
            "d": self.num_states,
            "empty_symbol": empty_sym_idx,
            "min_string_prob": min_string_prob,
            "max_string_length": max_string_length,
            "num_strings_to_find": num_strings_to_find,
            "depth_first": depth_first,
            "add_entropy": add_entropy,
            "disable_pbar": disable_pbar,
        }

        return params

    def _choose_next_state(
        self,
        curr_state: Node,
        random_state: {None, int, Iterable} = None,
        pred_method: str = "sample",
    ) -> SampledTransData:
        """
        Chooses the next state based on curr_state's transition distribution

        :param      curr_state:    The current state label
        :type       curr_state:    Hashable
        :param      random_state:  The np.random.RandomState() seed parameter
                                   for sampling from the state transition
                                   distribution. Defaulting to None causes the
                                   seed to reset.
        :param      pred_method:   The method used to choose the next state:
                                   'sample':
                                   sample from the transition
                                   distribution of the casual state of the
                                   automaton (the state the machine is left in
                                   after the sequence of observations). Makes
                                   non-deterministic predictions.
                                   'max_prob':
                                   like many language models, the selection of
                                   the next state s_{t+1}, and thus the next
                                   emitted symbol, conditioned on the set of
                                   observation symbols O_t = {o_1, ..., o_t}
                                   is:
                                   s_{t+1} = argmax_{s'}P(s' | s_t, O_t)
                                   makes deterministic predictions.
                                   {'sample', 'max_prob'}

        :returns:   The next state's label, the symbol emitted by changing
                    states, the probability of this transition occurring
        """

        trans_dist = self._get_node_data(curr_state, "trans_distribution")

        # critical step for use with parallelized libraries. This must be reset
        # before sampling, as otherwise each of the threads is using the same
        # seed, and we get lots of duplicated strings
        trans_dist.random_state = RandomState(random_state)

        # sampling an action (symbol) from the state-action distribution at
        # curr_state
        next_symbol_idx = trans_dist.rvs(size=1)[0]
        next_symbol = self._convert_symbol_idxs(next_symbol_idx)

        next_state, trans_probability = self._get_next_state(curr_state, next_symbol)
        return next_state, next_symbol, trans_probability

    def _get_next_state(
        self, curr_state: Node, symbol: Symbol
    ) -> Tuple[Node, Probability]:
        """
        Gets the next state given the current state and the "input" symbol.

        :param      curr_state:  The current state
        :param      symbol:      The input symbol

        :returns:   (The next state label, the transition probability)

        :raises     ValueError:  symbol not in curr_state's transition function
        :raises     ValueError:  duplicate symbol in curr_state's transition
                                 function
        """

        (possible_symbols, probabilities) = self._get_trans_probabilities(curr_state)

        if symbol not in possible_symbols:
            msg = (
                "given symbol ({}) is not found in the "
                "curr_state's ({}) "
                "transition distribution"
            ).format(symbol, curr_state)
            raise ValueError(msg)

        symbol_idx = [i for i, val in enumerate(possible_symbols) if val == symbol]
        num_matched_symbols = len(symbol_idx)
        if num_matched_symbols != 1:
            msg = (
                "given symbol ({}) is found multiple times in "
                "curr_state's ({}) "
                "transition distribution"
            ).format(symbol, curr_state)
            raise ValueError(msg)

        if self.is_sampleable:
            # stored in numpy array, so we just want the float probability
            # value
            symbol_probability = np.asscalar(probabilities[symbol_idx])
        else:
            symbol_probability = probabilities[symbol_idx[0]]

        next_state = self._transition_map[(curr_state, symbol)]

        return next_state, symbol_probability

    @staticmethod
    def _convert_states_edges(
        nodes: dict,
        edges: dict,
        final_transition_sym,
        empty_transition_sym,
        is_stochastic: bool,
    ) -> (bidict, NXNodeList, NXEdgeList):
        """
        Converts node and edges data from a manually specified YAML config file
        to the format needed by:
            - networkx.add_nodes_from()
            - networkx.add_edges_from()

        :param      nodes:                 dict of node objects to be converted
        :param      edges:                 dictionary adj. list to be converted
        :param      final_transition_sym:  representation of the termination /
                                           symbol
        :param      empty_transition_sym:  representation of the empty
                                           symbol (a.k.a. lambda)
        :param      is_stochastic:         the transitions are
                                           non-probabilistic, so we are going
                                           to assign a uniform distribution
                                           over all symbols for the purpose of
                                           generation

        :returns:   mapping to display symbols according to their
                    index in the transition distributions,
                    properly formated node and edge list containers
        :rtype:     tuple:
                    (symbol_display_map - bidirectional mapping of hashable
                                          symbols, to a unique integer index in
                                          the symbol map.
                     nodes - list of tuples:
                     (node label, node attribute dict),
                     edges - list of tuples:
                     (src node label, dest node label, edge attribute dict))
        """

        # need to convert the configuration adjacency list given in the config
        # to an edge list given as a 3-tuple of (source, dest, edgeAttrDict)
        edge_list = []
        seen_symbols = []

        # add these symbols first, so we can then later ensure they have the
        # last two indices
        seen_symbols.append(empty_transition_sym)
        seen_symbols.append(final_transition_sym)

        for source_node, dest_edges_data in edges.items():
            # don't need to add any edges if there is no edge data
            if dest_edges_data is None:
                continue

            for dest_node in dest_edges_data:
                symbols = dest_edges_data[dest_node]["symbols"]

                if is_stochastic:
                    probabilities = dest_edges_data[dest_node]["probabilities"]

                for symbol_idx, symbol in enumerate(symbols):
                    symbol = str(symbol)

                    # need to store new symbols in a map for display
                    if symbol not in seen_symbols:
                        seen_symbols.append(symbol)

                    edge_data = {"symbol": symbol}

                    if is_stochastic:
                        probability = probabilities[symbol_idx]
                        edge_data["probability"] = probability

                    newEdge = (source_node, dest_node, edge_data)
                    edge_list.append(newEdge)

        # best convention is to convert dict_items to a list, even though both
        # are iterable
        converted_nodes = list(nodes.items())

        # ensure that the empty and final symbols always have the last indices
        # in the display map for use in computations excluding those symbols
        symbol_display_map = bidict({})
        for new_sym_idx, symbol in enumerate(reversed(seen_symbols)):
            symbol_display_map[symbol] = new_sym_idx

        return symbol_display_map, converted_nodes, edge_list

    @abstractmethod
    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        Abstract method - must be overridden by subclass
        """

        raise NotImplementedError

    def _initialize_node_edge_properties(
        self,
        initial_weight_key: str = None,
        final_weight_key: str = None,
        state_observation_key: str = None,
        can_have_accepting_nodes: bool = True,
        edge_weight_key: str = None,
        merge_sinks: bool = False,
        **node_data_args: dict,
    ) -> None:
        """
        Initializes the node and edge data properties correctly.

        :param      initial_weight_key:        key in the automaton's node data
                                               corresponding to the weight /
                                               probability of starting in that
                                               node. If None, don't include
                                               this info in the display of the
                                               automaton.
        :param      final_weight_key:          key in the automaton's node data
                                               corresponding to the weight /
                                               probability of ending in that
                                               node. If None, don't include
                                               this info in the display of the
                                               automaton.
        :param      state_observation_key:     The key in each node's data dict
                                               for state observations. If None,
                                               don't include this info in the
                                               display of the automaton
        :param      can_have_accepting_nodes:  Indicates if the automata can
                                               have accepting nodes
        :param      edge_weight_key:           The key in each edge's data dict
                                               for edge weight / prob. If None,
                                               don't include this info in the
                                               display of the automaton
        :param      merge_sinks:               whether to combine all states
                                               together that have no outgoing
                                               edges
        :param      node_data_args:            keyword arguments to pass to
                                               _compute_node_data_properties()
        """

        if merge_sinks:
            self._nx_merge_sinks()

        # do batch computations at initialization, as these shouldn't
        # frequently change
        for node in self.nodes:
            self._compute_node_data_properties(node, **node_data_args)

        self._set_node_labels(
            initial_weight_key,
            final_weight_key,
            state_observation_key,
            can_have_accepting_nodes,
        )
        self._set_edge_labels(edge_weight_key)

        for state, symbol in self._transition_map.keys():
            self.symbols.add(symbol)
            self.state_labels.add(state)

        # final_transition_sym is just an internal not user-facing symbol
        if self.final_transition_sym in self.symbols:
            self.symbols.remove(self.final_transition_sym)

        # if we used smoothing, these might be larger, so we should expand them
        self.alphabet_size = len(self.symbols)
        self.num_states = len(self.state_labels)

        # not all automaton have observations, and this needs to be computed
        # after self.state_labels exists
        if self.num_obs is not None:
            for state in self.state_labels:
                self.observations.add(self.observe(state))

            N_actual_obs = len(self.observations)
            if N_actual_obs != self.num_obs:
                msg = (
                    f"given num_obs ({self.num_obs}) "
                    + "is different than the actual number of unique "
                    + f"observations seen ({N_actual_obs}) in the given "
                    + "graph data. proceeding using "
                    + f"self.num_obs = {N_actual_obs}."
                )
                warnings.warn(msg, RuntimeWarning)
            self.num_obs = N_actual_obs

        # wait until all node computations are done to make the vectorized rep.
        if self.is_stochastic:
            self._node_index_map = bidict(
                {state: index for index, state in enumerate(self.nodes)}
            )
            self._initial_state_distribution = self._make_initial_state_dist(
                self._node_index_map
            )
            self._final_state_distribution = self._make_final_state_dist(
                self._node_index_map
            )
            self._transition_matrices = self._make_transition_matrices(
                self._node_index_map
            )

    def _compute_node_data_properties(self, node: Node, **node_data_args: dict) -> None:
        """
        Base method for calculating the properties for the given node.

        :param      node:            The node to calculate properties for
        :param      node_data_args:  keyword arguments to
                                     _set_state_transition_dist

        :returns:   The node data properties.
        """
        # acceptance property shouldn't change after load in
        self._set_state_acceptance(node)

        # this edge key map is used to update all of the edges after
        # distribution setting
        self._edge_key_map.update(self._build_edge_key_map(node))

        # if we compute this once, we can sample from each distribution
        (edge_probs, edge_dests, edge_symbols) = self._set_state_transition_dist(
            node, edge_key_map=self._edge_key_map, **node_data_args
        )

        self._set_trans_map(node, edge_symbols, edge_dests)

    def _nx_merge_sinks(self) -> None:
        """
        merges all sink states (states with no outgoing edges)

        DOES not update all internal data structures, just the NX data
        """

        # Select all nodes with only 2 neighbors
        all_sinks = [n for n in self.nodes if len(list(self.neighbors(n))) == 0]

        if len(all_sinks) > 0:
            new_global_sink = all_sinks[0]

            for sink in all_sinks:
                if sink == new_global_sink:
                    continue

                curr_pred_nodes = list(self.predecessors(sink))
                job_queue = queue.Queue()
                for source in curr_pred_nodes:
                    edge_data = self.get_edge_data(source, sink).values()
                    old_edges = [(source, sink) for _ in edge_data]
                    new_edges = [(source, new_global_sink, data) for data in edge_data]

                    job_queue.put((edge_data, old_edges, new_edges))

                while not job_queue.empty():
                    edge_data, old_edges, new_edges = job_queue.get()

                    self.remove_edges_from(old_edges)
                    self.remove_nodes_from([edge[1] for edge in old_edges])
                    self.add_edges_from(new_edges)

    def _set_trans_map(
        self, curr_state: Node, edge_symbols: Symbols, edge_dests: Nodes
    ) -> None:
        """
        Sets the map of start state label and symbol to destination state

        :param      curr_state:  The current state label
        :param      edge_symbols:  The emitted symbols for each edge
        :param      edge_dests:    The labels of the destination states under
                                   each symbol at the curr_state

        :raises     ValueError:  checks for non-deterministic transitions
        """

        # creating the mapping from (start state, symbol) -> edge_dests
        disp_edge_symbols = self._convert_symbol_idxs(edge_symbols)
        state_symbol_keys = list(
            zip([curr_state] * len(disp_edge_symbols), disp_edge_symbols)
        )
        new_trans_map_entries = dict(zip(state_symbol_keys, edge_dests))

        # need to merge the newly computed transition map at node to the
        # existing map
        #
        # for a automaton, a given start state and symbol must have a
        # deterministic transition
        for key, dest_state in new_trans_map_entries.items():
            start_state = key[0]
            symbol = key[1]
            if key in self._transition_map:
                new_dest_state = dest_state != self._transition_map[key]
                same_start_state = key in self._transition_map
                non_deterministic_trans = same_start_state and new_dest_state
                if non_deterministic_trans:
                    msg = (
                        "duplicate transition from state {} "
                        "under symbol {} found - transition must be "
                        "deterministic"
                    ).format(start_state, symbol)
                    raise warnings.warn(msg)

                    self.is_deterministic = False

        self._transition_map = {**self._transition_map, **new_trans_map_entries}

    def _set_state_transition_dist(
        self,
        curr_state: Node,
        edge_key_map: dict,
        stochastic: {bool, None} = None,
        should_complete: {bool, None} = None,
        violating_state: {str, None} = None,
        complete: str = "smooth",
    ) -> Trans_data:
        """
        Sets the static state transition distribution for given state.

        :param      curr_state:       The current state label
        :param      edge_key_map:     mapping between all outgoing transitions
                                      and the networkx adjacency dictionary
                                      keys.
        :param      stochastic:       the transitions are non-probabilistic, so
                                      we are going to assign a uniform
                                      distribution over all symbols for the
                                      purpose of generation
        :param      should_complete:  Whether to try transition completion
        :param      violating_state:  The violating state name
        :param      complete:         Whether to ensure each transition is
                                      alphabet-complete.
                                      {'smooth', 'violate'}
                                      If 'smooth':
                                      The completeness processing will alter
                                      existing transition probabilities
                                      If 'violate':
                                      All completed states will be
                                      sent to the given violating state and the
                                      existing transition probability
                                      distributions will NOT be altered.

        :returns:   The new edge_probs, edge_dests, and edge_symbols added to
                    the underlying graph
        """

        # using class defaults if not given
        if stochastic is None:
            stochastic = self.is_stochastic
        if should_complete is None:
            should_complete = self._use_smoothing

        # need to convert the hashable symbols to their integer indices for
        # creating the categorical distribution, which only works with
        # integers
        edge_data = self.edges([curr_state], data=True)
        edge_dests = [edge[1] for edge in edge_data]

        original_edge_symbols = [edge[2]["symbol"] for edge in edge_data]
        edge_symbols = [
            self._symbol_display_map[symbol] for symbol in original_edge_symbols
        ]
        final_sym = self.final_transition_sym
        final_trans_symbol_idx = self._symbol_display_map[final_sym]

        if stochastic:
            # need to add final state probability to trans dist
            edge_probs = [edge[2]["probability"] for edge in edge_data]
            curr_final_state_prob = self._get_node_data(curr_state, "final_probability")

            # adding the final-state sequence end transition to the
            # distribution
            edge_probs.append(curr_final_state_prob)
            edge_dests.append(curr_state)
            edge_symbols.append(final_trans_symbol_idx)
        else:
            if self.is_sampleable:
                # using a uniform distribution to not bias the sampling of
                # symbols in a deterministic that does not actually have edge
                # probabilities
                num_symbols = len(edge_symbols)
                is_final_state = num_symbols == 0
                if is_final_state:
                    edge_probs = [1.0]
                    edge_dests.append(curr_state)
                    edge_symbols.append(final_trans_symbol_idx)
                else:
                    edge_probs = [1.0 / num_symbols for symbol in edge_symbols]
            else:
                edge_probs = [None]

        if should_complete:
            (edge_probs, edge_dests, edge_symbols) = self._complete_transitions(
                curr_state,
                edge_probs,
                edge_symbols,
                edge_dests,
                complete,
                violating_state,
            )

        if self.is_sampleable:
            if self.is_normalized:
                if sum(edge_probs) > 0.0:
                    edge_probs = [p / sum(edge_probs) for p in edge_probs]
                else:
                    edge_probs[-1] = 1.0

            next_sym_dist = rv_discrete(
                name="transition", values=(edge_symbols, edge_probs)
            )
        else:
            next_sym_dist = None

        # Need to update internal data structures with new node / edge data
        # This could have been changed from things like sample-ability /
        # stochasticity / completeness corrections.
        #
        # completing transitions handles edge updates internally
        new_disp_symbols = self._convert_symbol_idxs(edge_symbols)
        self._update_edges_from_lists(
            curr_state, edge_probs, new_disp_symbols, edge_dests, edge_key_map
        )
        self._set_node_data(curr_state, "trans_distribution", next_sym_dist)

        return edge_probs, edge_dests, edge_symbols

    def _complete_transitions(
        self,
        curr_state: Node,
        edge_probs: Probabilities,
        edge_symbols: Symbols,
        edge_dests: Nodes,
        complete: str = "smooth",
        dest_state: {Node, None} = None,
    ) -> Trans_data:
        """
        Computes missing transitions from the current state.

        This function will either:
        - apply Laplace smoothing to the given categorical state-symbol
          distributions as unlikely self-loops
        - add the missing transitions, but give the transitions no mass

        :param      curr_state:    The current state label for which to smooth
                                   the distribution
        :param      edge_probs:    The transition probability values for each
                                   edge
        :param      edge_symbols:  The emitted symbols for each edge
        :param      edge_dests:    The labels of the destination states under
                                   each symbol at the curr_state
        :param      complete:      Whether to ensure each transition is
                                   alphabet-complete.
                                   {'smooth', 'violate'}
                                   If 'smooth':
                                   The completeness processing will alter
                                   existing transition probabilities
                                   If 'violate':
                                   All completed states will be sent to the
                                   given violating state and the existing
                                   transition probability distributions will
                                   NOT be altered.
        :param      dest_state:    The destination state label for the missing
                                   transitions.
                                   (default curr_state)

        :returns:   The smoothed / completed version of edge_probs, edge_dests,
                    and edge_symbols

        :raises     ValueError:    Invalid setting of complete
        :raises     ValueError:    using 'violate' completeness setting, but
                                   no violating destination state name given
        :raises     ValueError:    Too-large setting of self._smoothing_amount
                                   results in laplace smoothing being
                                   impossible
        """

        # need to check and set completion algorithm
        allowed_completion_algs = ["smooth", "violate"]

        if complete not in allowed_completion_algs:
            msg = (
                f"given complete setting ({complete}) is not one: "
                + f"{allowed_completion_algs}"
            )
            raise ValueError(msg)

        if complete == "violate" and not dest_state:
            msg = (
                f"if using the {complete} setting, you must provide "
                + "a violating state label to send added transitions to."
            )
            raise ValueError(msg)
        elif complete == "smooth" and not dest_state:
            dest_state = curr_state

        # setting the amount of probability mass to add to completed
        # transitions
        if complete == "smooth":
            prob_to_add = self._smoothing_amount
        elif complete == "violate":
            # no probability of transition to the violating state,
            # but the violating state must have a uniform self-transition
            # distribution over all possible symbols, except for the
            # "termination" symbol - violating state never terminates.
            if dest_state == curr_state:
                prob_to_add = 1.0 / self.alphabet_size
            else:
                prob_to_add = 0.0

        # here we add in the missing transition probabilities as just very
        # unlikely self-loops ('smooth') or 0 probability transitions to the
        # violating state ('violate')
        num_of_missing_transitions = 0
        new_edge_probs, new_edge_dests, new_edge_symbols = [], [], []
        all_symbols_idxs = list(self._symbol_display_map.inv.keys())

        # actually creating the completed transitions
        for symbol in all_symbols_idxs:
            if symbol not in edge_symbols:
                num_of_missing_transitions += 1
                new_edge_probs.append(prob_to_add)
                new_edge_dests.append(dest_state)
                new_edge_symbols.append(symbol)

        # re-arranging probability mass in the case of needing smoothing
        if complete == "smooth":
            all_possible_trans = [
                idx for idx, prob in enumerate(edge_probs) if prob > 0.0
            ]
            num_orig_samples = len(all_possible_trans)

            # now, we need to remove the smoothed probability mass from the
            # original transition distribution
            num_added_symbols = len(new_edge_symbols)
            added_prob_mass = self._smoothing_amount * num_added_symbols
            smoothing_per_orig_trans = added_prob_mass / num_orig_samples

            for trans_idx in all_possible_trans:
                if edge_probs[trans_idx] < smoothing_per_orig_trans:
                    msg = (
                        "smoothing failed: transition from state "
                        + f"{curr_state} to state {edge_dests[trans_idx]} "
                        + f"under symbol {edge_symbols[trans_idx]} has "
                        + "too little probability mass "
                        + f"({edge_probs[trans_idx]}) to distribute the "
                        + "desired amount of per-symbol smoothing "
                        + f"(self._smoothing_amount = {prob_to_add})"
                    )
                    raise ValueError(msg)

                edge_probs[trans_idx] -= smoothing_per_orig_trans

        # combining the new transitions with the smoothed, original
        # distribution to get the final smoothed distribution
        edge_probs += new_edge_probs
        edge_dests += new_edge_dests
        edge_symbols += new_edge_symbols

        return edge_probs, edge_dests, edge_symbols

    def _make_initial_state_dist(self, node_index_map: bidict) -> np.ndarray:
        """
        Creates the pmf for the initial state distribution as a numpy array.

        :param      node_index_map:  The mapping from state label to index in
                                     vectorized representation of the
                                     distribution

        :returns:   (1 x num_states) numpy array containing the probability
                    distribution of starting at each state's index
        """

        start_state_index = node_index_map[self.start_state]
        initial_state_distribution = np.zeros(shape=(1, self.num_states))
        initial_state_distribution[0, start_state_index] = 1.0

        return initial_state_distribution

    def _make_final_state_dist(self, node_index_map: bidict) -> np.ndarray:
        """
        Creates the pmf for the final state distribution as a numpy array.

        :param      node_index_map:  The mapping from state label to index in
                                     vectorized representation of the
                                     distribution

        :returns:   (num_states x 1) numpy array containing the probability
                    distribution of terminating at each state's index
        """

        final_state_distribution = np.zeros(shape=(self.num_states, 1))

        for node, node_index in node_index_map.items():
            final_prob = self._get_node_data(node, "final_probability")
            final_state_distribution[node_index, 0] = final_prob

        return final_state_distribution

    def _make_transition_matrices(self, node_index_map: bidict) -> dict:
        """
        Creates the mapping from a symbol to the state transition matrix under
        the given symbol.

        Not necessarily a proper stochastic matrix, especially in the case of
        stochastic matrices. Should be properly stochastic if is_sampleable.

        :param      node_index_map:  The mapping from state label to index in
                                     vectorized representation of the
                                     distribution

        :returns:   mapping from each symbol to the (num_states x num_states)
                    numpy matrix containing the probability of transitioning
                    to state i to state j under the given symbol at entry [i,j]
        """

        weight = "probability"
        nonedge_trans_prob = 0.0
        nodelist = list(node_index_map)
        nodeset = set(node_index_map)

        if len(nodelist) != len(nodeset):
            msg = "Ambiguous ordering: `nodelist` contained duplicates."
            raise nx.NetworkXError(msg)
        transition_matrices = dict()

        for symbol in self.symbols:
            A = np.full((self.num_states, self.num_states), np.nan)
            transition_matrices[symbol] = copy.deepcopy(A)

        for u, v, attrs in self.edges(data=True):
            symbol = attrs["symbol"]
            curr_trans_mat = transition_matrices[symbol]

            if (u in nodeset) and (v in nodeset):
                i, j = node_index_map[u], node_index_map[v]
                e_weight = attrs.get(weight, 1)
                curr_trans_mat[i, j] = e_weight

            transition_matrices[symbol] = curr_trans_mat

        for symbol in self.symbols:
            A = transition_matrices[symbol]
            A[np.isnan(A)] = nonedge_trans_prob
            A = np.asarray(A)

        return transition_matrices

    def _convert_symbol_idxs(self, integer_symbols: {List[int], int}) -> List:
        """
        Convert an iterable container of integer representations of automaton
        symbols to their readable, user-meaningful form.

        :param      integer_symbols:  The integer symbol(s) to convert

        :returns:   a list of displayable automaton symbols corresponding to
                    the inputted integer symbols

        :raises     ValueError:       all given symbol indices must be ints
        """

        display_symbols = []

        # need to do type-checking / polymorphism handling here
        if not isinstance(integer_symbols, collections.abc.Iterable):
            if isinstance(integer_symbols, int):
                return self._symbol_display_map.inv[integer_symbols]
            elif type(integer_symbols).__module__ == "numpy":
                if np.issubdtype(integer_symbols, np.integer):
                    return self._symbol_display_map.inv[integer_symbols]
            else:
                msg = f"symbol index ({integer_symbols}) is not an int"
                raise ValueError(msg)
        else:
            all_ints = all(
                np.issubdtype(type(sym), np.integer) for sym in integer_symbols
            )
            if not all_ints:
                msg = f"not all symbol indices ({integer_symbols}) are ints"
                raise ValueError(msg)

        for integer_symbol in integer_symbols:
            converted_symbol = self._symbol_display_map.inv[integer_symbol]
            display_symbols.append(converted_symbol)

        return display_symbols

    def _get_pydot_representation(self) -> Dot:
        """
        converts the networkx graph to pydot and sets graphviz graph attributes

        :returns:   The pydot Dot data structure representation.
        :rtype:     pydot.Dot
        """

        graph = to_pydot(self)
        graph.set_splines(True)
        graph.set_nodesep(0.5)
        graph.set_sep("+25,25")
        graph.set_ratio(1)

        return graph

    def _build_edge_key_map(self, curr_state: Node) -> dict:
        """
        Builds a mapping between all outgoing transitions and the
        networkx adjacency dictionary keys.

        the mapping M maps:

        (current node, symbol, destination node) -> edge key in current
                                                    node's adj dict

        :param      curr_state:  The node label to build the mapping at

        :returns:   The edge key map.
        """

        trans_to_edge_key_map = dict()
        for dest_state, edges in self[curr_state].items():
            for edge_key, edge_data in edges.items():
                trans = (curr_state, edge_data["symbol"], dest_state)
                trans_to_edge_key_map[trans] = edge_key

        return trans_to_edge_key_map

    def _get_trans_probabilities(
        self, curr_state: Node
    ) -> Tuple[Symbols, Probabilities]:
        """
        Extracts the transition probabilities and associated symbols at the
        current state.

        :param      curr_state:  The curr state

        :returns:   The transition probabilities and associated symbols
        """

        if self.is_sampleable:
            trans_distribution = self._get_node_data(curr_state, "trans_distribution")
            possible_symbols = self._convert_symbol_idxs(trans_distribution.xk)
            probabilities = trans_distribution.pk
        else:
            possible_symbols = [
                symbol
                for (state, symbol) in self._transition_map.keys()
                if state == curr_state
            ]
            possible_symbols = [
                edge["symbol"]
                for state, edges in self[curr_state].items()
                for key, edge in edges.items()
            ]
            if self.is_stochastic:
                probabilities = [
                    edge["probability"]
                    for state, edges in self[curr_state].items()
                    for key, edge in edges.items()
                ]
            else:
                probabilities = [None for i in range(len(possible_symbols))]

        return possible_symbols, probabilities

    @staticmethod
    def _get_abbadingo_string(
        trace: Symbols, trace_length: int, is_pos_example: bool
    ) -> str:
        """
        Returns the Abbadingo (sigh) formatted string given a trace string and
        the label for the trace

        :param      trace:           The trace string to represent in Abbadingo
        :param      trace_length:    The trace length
        :param      is_pos_example:  Indicates if the trace is a positive
                                     example of the pdfa

        :returns:   The abbadingo formatted string for the given trace
        """

        trace = " ".join(str(x) for x in trace)
        trace_label = {False: "0", True: "1"}[is_pos_example]

        return trace_label + " " + str(trace_length) + " " + trace + "\n"

    def _set_node_labels(
        self,
        initial_weight_key: str,
        final_weight_key: str,
        state_observation_key: str,
        can_have_accepting_nodes: bool,
        graph: {None, nx.MultiDiGraph} = None,
    ) -> None:
        """
        Sets each node's label property for use in graphviz output

        :param      initial_weight_key:        key in the automaton's node data
                                               corresponding to the weight /
                                               probability of starting in that
                                               node. If None, don't include
                                               this info in the display of the
                                               automaton.
        :param      final_weight_key:          key in the automaton's node data
                                               corresponding to the weight /
                                               probability of ending in that
                                               node. If None, don't include
                                               this info in the display of the
                                               automaton.
        :param      state_observation_key:     The state observation key
        :param      can_have_accepting_nodes:  Indicates if the automata can
                                               have accepting nodes
        :param      graph:                     The graph to access.
                                               Default = None => use instance
        :type       graph:                     {None, nx.MultiDiGraph}
        :type       final_weight_key:          string
        :type       can_have_accepting_nodes:  boolean
        """

        if graph is None:
            graph = self

        label_dict = {}

        for node_name, node_data in graph.nodes.data():
            nodel_key = node_name

            if initial_weight_key is not None:
                weight = node_data[initial_weight_key]
                initial_wt_string = edge_weight_to_string(weight)
                node_name = initial_wt_string + " : " + node_name
            else:
                node_dot_label_string = node_name

            if final_weight_key is not None:
                weight = node_data[final_weight_key]
                final_wt_string = edge_weight_to_string(weight)
                node_dot_label_string = node_name + " : " + final_wt_string
            else:
                node_dot_label_string = node_name

            graphviz_node_label = {
                "label": node_dot_label_string,
                "fillcolor": "gray80",
                "style": "filled",
            }

            if state_observation_key is not None:
                obs_label = node_obs_to_str(node_data[state_observation_key])
                external_label = "{" + obs_label + "}"
                graphviz_node_label["xlabel"] = external_label

            is_start_state = nodel_key == self.start_state

            # colors are ranked in increasing importance
            if "color" in node_data:
                color = node_data["color"]
                graphviz_node_label.update({"fillcolor": color})

            if "is_violating" in node_data and node_data["is_violating"]:
                graphviz_node_label.update({"shape": "diamond"})
                graphviz_node_label.update({"fillcolor": "tomato1"})

            if "is_accepting" in node_data and node_data["is_accepting"]:
                graphviz_node_label.update({"peripheries": 2})
                graphviz_node_label.update({"shape": "doubleoctagon"})
                graphviz_node_label.update({"fillcolor": "lawngreen"})

            if is_start_state:
                graphviz_node_label.update({"shape": "box"})
                graphviz_node_label.update({"fillcolor": "royalblue1"})

            label_dict[nodel_key] = graphviz_node_label

        nx.set_node_attributes(graph, label_dict)

    def _set_edge_labels(
        self, edge_weight_key: str = None, graph: {None, nx.MultiDiGraph} = None
    ) -> None:
        """
        Sets each edge's label property for use in graphviz output

        :param      edge_weight_key:  The edge data's "weight" key
        :param      graph:            The graph to access.
                                      Default = None => use instance
        """

        if graph is None:
            graph = self

        # this needs to be a mapping from edges (node label tuples) to a
        # dictionary of attributes
        label_dict = {}

        for u, v, key, data in graph.edges(data=True, keys=True):
            if edge_weight_key is not None:
                wt_str = edge_weight_to_string(data[edge_weight_key])
                edge_label_string = str(data["symbol"]) + ": " + wt_str
            else:
                edge_label_string = str(data["symbol"])

            new_label_property = {"label": edge_label_string, "fontcolor": "blue"}
            node_identifier = (u, v, key)

            label_dict[node_identifier] = new_label_property

        nx.set_edge_attributes(graph, label_dict)

    def _get_node_data(
        self, node_label: Node, data_key: str, graph: {None, nx.MultiDiGraph} = None
    ):
        """
        Gets the node's data_key data from the graph

        :param      node_label:  The node label
        :param      data_key:    The desired node data's key name
        :param      graph:       The graph to access. Default = None => use
                                 instance

        :returns:   The node data associated with the node_label and data_key
        :rtype:     type of self.nodes.data()[node_label][data_key]
        """

        if graph is None:
            graph = self

        node_data = graph.nodes.data()

        if not self.is_sampleable and data_key == "trans_distribution":
            msg = (
                "automaton is not sampleable and thus does not have "
                + "transition distributions"
            )
            raise TypeError(msg)

        return node_data[node_label][data_key]

    def _set_node_data(
        self,
        node_label: Node,
        data_key: str,
        data,
        graph: {None, nx.MultiDiGraph} = None,
    ) -> None:
        """
        Sets the node's data_key data from the graph

        :param      node_label:  The node label
        :param      data_key:    The desired node data's key name
        :param      data:        The data to associate with data_key
        :param      graph:       The graph to access.
                                 Default = None => use instance
        """

        if graph is None:
            graph = self

        node_data = graph.nodes.data()
        node_data[node_label][data_key] = data

    def _get_edge_data(
        self,
        src_node_label: Node,
        dest_node_label: Node,
        graph: {None, nx.MultiDiGraph} = None,
    ) -> dict:
        """
        Gets all edge between src and dest's data dicts from the graph

        :param      src_node_label:   The edge's source node edge label
        :param      dest_node_label:  The edge's destination node label
        :param      graph:            The graph to access.
                                      Default = None => use instance

        :returns:   The edge data dict associated with the src and dest labels
                    and the desired data_key
        """

        if graph is None:
            graph = self

        edge_data = dict(graph[src_node_label][dest_node_label])

        return edge_data

    def _set_edge_data(
        self,
        src_node_label: Node,
        dest_node_label: Node,
        data: dict,
        graph: {None, nx.MultiDiGraph} = None,
    ) -> None:
        """
        Sets the edge's data_key with given data

        :param      src_node_label:   The edge's source node edge label
        :param      dest_node_label:  The edge's destination node label
        :param      data:             The data to associate with data_key
        :param      graph:            The graph to access.
                                      Default = None => use instance
        """

        if graph is None:
            graph = self

        graph[src_node_label][dest_node_label] = data

    def _update_edges(
        self,
        node_label: Node,
        new_edge_data: Dict[Node, Dict[Symbol, Dict]],
        graph: {None, nx.MultiDiGraph} = None,
    ) -> None:
        """
        Updates / adds any edges to the graph given new edge data at node

        :example
        new_edge_data = {dest_node_label: {'sym1': {'probability': 0.35},
                                           'sym2': {'probability': 0.65}}}

        :param      node_label:     The node label
        :param      new_edge_data:  The labels of the destination states under
                                    each symbol at the curr_state
        :param      graph:          The graph to access.
                                    Default = None => use instance
        """

        if graph is None:
            graph = self

        adj = {node_label: new_edge_data}

        e = [
            (u, v, ekey, d)
            for u, nbrs in adj.items()
            for v, keydict in nbrs.items()
            for ekey, d in keydict.items()
        ]

        graph.update(edges=e)

    def _update_edges_from_lists(
        self,
        curr_state: Node,
        edge_probs: Probabilities,
        edge_symbols: Symbols,
        edge_dests: Nodes,
        edge_key_map: dict,
    ) -> None:
        """
        Updates edge data given lists of new edge attributes

        :param      curr_state:    The current state label for which to smooth
                                   the distribution
        :param      edge_probs:    The transition probability values for each
                                   edge
        :param      edge_symbols:  The emitted symbols for each edge
        :param      edge_dests:    The labels of the destination states under
                                   each symbol at the curr_state
        :param      edge_key_map:  mapping between all outgoing
                                   transitions and the networkx adjacency
                                   dictionary keys.
        """

        transitions = zip(edge_dests, edge_symbols, edge_probs)

        new_edges = []
        for dest_state, symbol, prob in transitions:
            trans = (curr_state, symbol, dest_state)

            # for some reason, MultiGraph dictionaries can't check any form of
            # duplicate edges, so we have to manually check here :(
            if trans in edge_key_map:
                trans_key = edge_key_map[trans]
                self.remove_edge(curr_state, dest_state, key=trans_key)

            # final transitions are handled by the node's final probability,
            # so don't manually add these transitions
            if not symbol == self.final_transition_sym:
                edge_data = {"symbol": symbol, "probability": prob}
                new_edges.append((curr_state, dest_state, edge_data))

        self.add_edges_from(new_edges)


def node_obs_to_str(obs: Observation) -> str:
    """
    returns a node observation label as an appropriately formatted string

    :param      obs:  The node observation label

    :returns:   properly formatted observation label string
    """
    if isinstance(obs, int):
        obs_str = "{obs:d}".format(obs=obs)
    elif isinstance(obs, str):
        obs_str = obs
    else:
        msg = f"obs ({obs} of type ({type(obs)}) must be of type: int, str)"
        raise ValueError(msg)

    return obs_str


def edge_weight_to_string(weight: {int, float}) -> str:
    """
    returns a numeric edge weight as an appropriately formatted string

    :param      weight:  The edge weight to convert to string.
    :type       weight:  int or float

    :returns:   properly formatted weight string
    :rtype:     string
    """
    if isinstance(weight, int):
        wt_str = "{weight:d}".format(weight=weight)
    elif isinstance(weight, float):
        wt_str = "{weight:.{digits}f}".format(weight=weight, digits=2)

    return wt_str
