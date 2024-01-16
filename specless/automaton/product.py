import copy
import math
import queue
import re
import warnings
from collections.abc import Iterable
from typing import Tuple

from bidict import bidict
from scipy.stats import rv_discrete

# local packages
from specless.factory.builder import Builder

from .base import Automaton
from .pdfa import PDFA
from .transition_system import TransitionSystem
from .types import (
    GeneratedTraceData,
    Node,
    NXEdgeList,
    NXNodeList,
    Observation,
    Probability,
    Symbol,
    Symbols,
)
from .utils import MaxHeap

# define these type defs for method annotation type hints
TS_Trans_Data = Tuple[Node, Observation]

IS_STOCHASTIC = True
SPEC_VIOLATING_STATE = "q_v"


class Product(Automaton):

    """
    Describes a product automaton between a specification automaton
    and a dynamics automaton.

    You can use this class to compose the two automaton together and then find
    a controller for the dynamical system that satisfies the specification

        :param      nodes:                          node list as expected by
                                                    networkx.add_nodes_from()
                                                    (node label, node attribute
                                                    dict)
        :param      edges:                          edge list as expected by
                                                    networkx.add_edges_from()
                                                    (src node label, dest node
                                                    label, edge attribute dict)
        :param      symbol_display_map:             bidirectional mapping of
                                                    hashable symbols, to a
                                                    unique integer index in the
                                                    symbol map. Needed to
                                                    translate between the
                                                    indices in the transition
                                                    distribution and the
                                                    hashable representation
                                                    which is meaningful to the
                                                    user
        :param      alphabet_size:                  number of symbols in system
                                                    alphabet
        :param      num_states:                     number of states in
                                                    automaton state space
        :param      start_state:                    unique start state string
                                                    label of system
        :param      num_obs:                        number of observation
                                                    symbols
        :param      final_transition_sym:           representation of the
                                                    termination symbol. If not
                                                    given, will default to base
                                                    class default.
        :param      empty_transition_sym:           representation of the empty
                                                    symbol (a.k.a. lambda). If
                                                    not given, will default to
                                                    base class default.
        :param      is_normalized:                  whether to renormalize the
                                                    edge probabilities such
                                                    that each states has a well
                                                    defined transition
                                                    probability distribution.
                                                    We typically DONT want to
                                                    modify the probabilities of
                                                    the product algorithm,
                                                    except if we would like to
                                                    be able to easily sample
                                                    traces
    """

    def __init__(
        self,
        nodes: NXNodeList,
        edges: NXEdgeList,
        symbol_display_map: bidict,
        alphabet_size: int,
        num_states: int,
        start_state: Node,
        num_obs: int,
        final_transition_sym: Symbol,
        empty_transition_sym: Symbol,
        is_normalized: bool,
    ) -> "Product":
        """
        Constructs a new instance of an Product automaton object.
        """

        # if we normalize the probabilities
        if is_normalized:
            is_sampleable = True
        else:
            is_sampleable = False

        # need to start with a fully initialized automaton
        super().__init__(
            nodes,
            edges,
            symbol_display_map,
            alphabet_size,
            num_states,
            start_state,
            smooth_transitions=False,
            is_stochastic=IS_STOCHASTIC,
            is_sampleable=is_sampleable,
            is_normalized=is_normalized,
            num_obs=num_obs,
            final_transition_sym=final_transition_sym,
            empty_transition_sym=empty_transition_sym,
            final_weight_key="final_probability",
            state_observation_key="observation",
            can_have_accepting_nodes=True,
            edge_weight_key="probability",
        )

    def compute_strategy(
        self,
        min_string_probability: {Probability, None} = None,
        max_string_length: {int, None} = None,
    ) -> Tuple[Symbols, Probability]:
        """
        Calculates a control strategy for the dynamical system that best
        matches the language of the specification.

        :param      min_string_probability:  The minimum string probability
        :param      max_string_length:       The maximum string length

        :returns:   The sequence of controls to apply, the probability in the
                    languge of the specification of generating the output word
                    of the dynamical system under the control symbols.
        """

        # using DFS for BMPS as products are often very deep, tree-like
        (controls_symbols, obs_prob, _) = self.most_probable_string(
            min_string_probability, max_string_length, depth_first=True
        )

        # None -> completely incompatible
        if controls_symbols is None:
            msg = (
                "no valid controller possible for given settings of "
                + f"min_string_probability {min_string_probability} and "
                + f"max_string_length {max_string_length}."
            )
            warnings.warn(msg, RuntimeWarning)

            return controls_symbols, obs_prob

        if len(controls_symbols) == 1:
            msg = (
                'only "initialization" symbol artificially added to '
                + "the TS found to be most probable controller -> "
                + "specification and dynamical system are "
                + "incompatible. Try adjusting "
                + f"min_string_probability {min_string_probability} and "
                + f"max_string_length {max_string_length}, or trying a "
                + "different solver."
            )
            warnings.warn(msg, RuntimeWarning)

        # as seen above, first symbol is a useless / artifical initialization
        # symbol
        if isinstance(controls_symbols, Iterable):
            controls_symbols = controls_symbols[1:]

        return controls_symbols, obs_prob

    def generate_traces(
        self,
        num_samples: int,
        N: int = None,
        num_traces_to_find: int = None,
        min_trace_probability: Probability = None,
        complete_samples: bool = False,
        max_resamples: int = 100,
        use_greedy_MPS_sampler: bool = True,
        force_MPS_sampler: bool = False,
        return_whatever_you_got: bool = False,
        force_multicore: bool = True,
        show_progress_bar: bool = True,
    ) -> GeneratedTraceData:
        """
        Tries to generate num_samples random traces from the product.

        :param      num_samples:              The number of trace samples to
                                              generate
        :param      N:                        maximum length of any trace
        :param      num_traces_to_find:       the number of base random traces
                                              to find in the automaton when
                                              using the MPS sampler. This is
                                              not necessarily the same as
                                              num_samples, as often the MPS
                                              sampler is too slow to return
                                              that many samples. Thus, if you
                                              allow for it, the MPS samples can
                                              be resampled to return
                                              num_samples samples after the MPS
                                              sampler is done.
        :param      min_trace_probability:    The minimum trace probability.
                                              only needed when using the MPS
                                              sampler. Lowering this will
                                              result in more random (less
                                              representative traces), but will
                                              make the algorithm much faster.
                                              If set TOO high, you will find no
                                              traces meeting this requirement.
                                              (default 0.0)
        :param      complete_samples:         If enabled, if the underlying
                                              sampler fails to generate
                                              num_samples, then any samples it
                                              does find will be resampled to
                                              create num_samples samples
                                              occurring with correct RELATIVE
                                              frequencies.
        :param      max_resamples:            The maximum number of times to
                                              resample if if we create a trace
                                              of length N that still doesn't
                                              have a probability > 0 in the
                                              language
        :param      use_greedy_MPS_sampler:   whether to try using the MUCH
                                              faster greedy search algorithm.
                                              only possible if the automaton
                                              has deterministic transitions.
                                              Only set this to False if the
                                              automaton actually is
                                              non-deterministic, as the
                                              non-deterministic solver is an
                                              approximation and MUCH slower.
        :param      force_MPS_sampler:        by default, IF THE PRODUCT HAS
                                              BEEN NORMALIZED (and thus is
                                              sampleable), then it will fall
                                              back on the base class' sampler.
                                              Not available if the product is
                                              not is_sampleable. This sampler
                                              is truly a random MC sampler, and
                                              thus is appropriate if you want
                                              to generate traces with more
                                              randomness than the MPS sampler.
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
        :param      show_progress_bar:        whether to show a tqdm progress
                                              bar for each sampled trace. Only
                                              turn this on if you're sampling a
                                              few traces from a very
                                              expensive-to-sample automaton.

        :returns:   list of sampled traces, list of the associated trace
                    lengths, list of the associated trace probabilities
        :rtype:     tuple(list(list(int)), list(int), list(float))
        """

        if not use_greedy_MPS_sampler and N is None:
            msg = (
                "Must provide a value for N if not using the "
                + "use_greedy_MPS_sampler"
            )
            raise ValueError(msg)

        if self.is_sampleable and not force_MPS_sampler:
            results = super().generate_traces(
                num_samples=num_samples,
                N=N,
                max_resamples=max_resamples,
                return_whatever_you_got=return_whatever_you_got,
                force_multicore=force_multicore,
            )

            controls, _, sequence_probs = results
            if controls is not None:
                # convert to max heap to match MPS sampling returns
                viable_traces = MaxHeap()
                for prob, control in zip(sequence_probs, controls):
                    viable_traces.heappush((prob, control))
            else:
                viable_traces = None

        else:
            # making sure that these params were provided if using MPS sampler
            if num_traces_to_find is None and not use_greedy_MPS_sampler:
                num_traces_to_find = num_samples

                msg = (
                    "No value given for num_traces_to_find. Using "
                    + "default value of num_traces_to_find = num_samples "
                    + f"({num_samples}). This may be too slow for large "
                    + "product automata. Try providing a value of "
                    + "num_traces_to_find less than num_samples and "
                    + "enabling complete_samples"
                )
                warnings.warn(msg)

            if min_trace_probability is None and not use_greedy_MPS_sampler:
                min_trace_probability = 0.0

                msg = (
                    "No value given for min_trace_probability. Using "
                    + f"default value of {min_trace_probability}"
                )
                warnings.warn(msg)

            if use_greedy_MPS_sampler:
                _, _, viable_traces = self.most_probable_string(
                    try_to_use_greedy=use_greedy_MPS_sampler
                )
            else:
                # we want to sample from both the most and least likely traces
                # to get some diversity for resampling
                N_DFS = math.ceil(num_traces_to_find / 2)
                _, _, viable_traces_min = self.most_probable_string(
                    min_string_probability=min_trace_probability,
                    max_string_length=N,
                    num_strings_to_find=N_DFS,
                    try_to_use_greedy=False,
                    backwards_search=True,
                    depth_first=True,
                    add_entropy=True,
                )
                N_BFS = math.ceil(num_traces_to_find / 2)
                _, _, viable_traces_max = self.most_probable_string(
                    min_string_probability=min_trace_probability,
                    max_string_length=N,
                    num_strings_to_find=N_BFS,
                    try_to_use_greedy=False,
                    backwards_search=True,
                    depth_first=False,
                    add_entropy=True,
                )

                # merge the two heaps
                viable_traces = MaxHeap()
                for item_1 in viable_traces_min:
                    viable_traces.heappush(item_1)
                for item_2 in viable_traces_max:
                    viable_traces.heappush(item_2)

        # need to post-process the sampled data, as this is a product
        if viable_traces is not None:
            # resampling the viable traces to ensure we always have num_samples
            # samples
            if len(viable_traces) < num_samples and complete_samples:
                probs, symbols = zip(*viable_traces)

                # need to normalize the probabilities, as we won't have the
                # full trace distribution
                normalized_probs = [prob / sum(probs) for prob in probs]
                trace_idxs = list(range(len(symbols)))

                trace_dist = rv_discrete(values=(trace_idxs, normalized_probs))
                sampled_trace_idxs = trace_dist.rvs(size=num_samples)

                new_traces = [symbols[idx] for idx in sampled_trace_idxs]
                new_probs = [probs[idx] for idx in sampled_trace_idxs]

                viable_traces = MaxHeap()
                for prob, trace in zip(new_probs, new_traces):
                    viable_traces.heappush((prob, trace))

            samples, trace_lengths, trace_probs = [], [], []

            for prob, controls in viable_traces:
                if controls is not None:
                    # the first control symbol is always the initialization
                    # symbol, so we should remove it for general use
                    if isinstance(controls, Iterable):
                        sample = controls[1:]
                    else:
                        sample = controls

                    sample_length = len(sample)
                else:
                    sample = controls
                    sample_length = None

                samples.append(sample)
                trace_lengths.append(sample_length)
                trace_probs.append(prob)
        else:
            samples, trace_lengths, trace_probs = None, None, None

        return samples, trace_lengths, trace_probs

    def observe(self, curr_state: Node) -> Observation:
        """
        Returns the given state's observation symbol

        :param      curr_state:  The current product state

        :returns:   observation symbol emitted at curr_state
        """

        return self._get_node_data(curr_state, "observation")

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        If curr_state's final_probability == 1.00 then the state is guaranteed
        to be final

        :param      curr_state:  The current state's node label
        """

        curr_final_prob = self._get_node_data(curr_state, "final_probability")

        if curr_final_prob >= 1.00:
            state_accepts = True
        else:
            state_accepts = False

        self._set_node_data(curr_state, "is_accepting", state_accepts)

    @classmethod
    def _complete_specification(cls, specification: PDFA) -> PDFA:
        """
        processes the automaton and makes sure each state has a transition for
        each symbol

        completed nodes will be sent to "violating", cyclic state with
        uniform probability over all symbols, as producing the missing symbols
        is impossible given the language defined by the specification

        :param      specification:  The specification to complete

        :returns:   the completed version of the specification
        """

        # first need to define and add the "violating" state to the
        # specification's underlying graph
        violating_state = SPEC_VIOLATING_STATE
        violating_state_props = {
            "final_probability": 0.00,
            "trans_distribution": None,
            "is_accepting": None,
            "is_violating": True,
        }
        specification.add_node(violating_state, **violating_state_props)

        specification._initialize_node_edge_properties(
            final_weight_key="final_probability",
            can_have_accepting_nodes=True,
            edge_weight_key="probability",
            should_complete=True,
            violating_state=violating_state,
            complete="violate",
        )

        return specification

    @classmethod
    def _augment_initial_state(
        cls, dynamical_system: TransitionSystem, specification: PDFA
    ) -> TransitionSystem:
        """
        Adds an initialization state to the dynamical system to maintain
        language distributional similarity with the specification

        :param      dynamical_system:  The dynamical system to augment
        :param      specification:     The specification to take a product with

        :returns:   The transition system with a new initialization state added
        """

        initialization_state = "x_init"

        spec_empty_symbol = specification.empty_transition_sym
        initialization_state_props = {"observation": spec_empty_symbol}
        if spec_empty_symbol not in dynamical_system.observations:
            dynamical_system.observations.add(spec_empty_symbol)
            dynamical_system.num_obs += 1

        dynamical_system.add_node(initialization_state, **initialization_state_props)
        old_start_state = dynamical_system.start_state
        dynamical_system.start_state = initialization_state

        # can choose any symbol to be the initialization symbol, doesn't matter
        initialization_symbol = list(dynamical_system.symbols)[0]
        initialization_edge_props = {
            "symbol": initialization_symbol,
            "probability": 1.00,
        }
        dynamical_system.add_edge(
            initialization_state, old_start_state, **initialization_edge_props
        )

        dynamical_system._initialize_node_edge_properties(
            can_have_accepting_nodes=False,
            state_observation_key="observation",
            should_complete=False,
        )

        return dynamical_system

    @classmethod
    def _compute_product(
        cls, dynamical_system: TransitionSystem, specification: PDFA
    ) -> dict:
        """
        Calculates the product automaton given pre-processed automata

        :param      dynamical_system:  The dynamical system
        :param      specification:     The specification

        :returns:   The initialized product automaton configuration data
        """

        # naming to follow written algorithm
        T = dynamical_system
        A = specification
        Sigma = dynamical_system.symbols

        x_init = T.start_state
        q_init = A.start_state
        init_prod_state = cls._get_product_state_label(x_init, q_init)

        nodes = {}
        edges = {}

        search_queue = queue.Queue()
        search_queue.put((x_init, q_init))
        visited = set()

        while not search_queue.empty():
            x, q = search_queue.get()

            for sigma in Sigma:
                dyn_trans = (x, sigma)
                dynamically_compatible = dyn_trans in T._transition_map

                if not dynamically_compatible:
                    specification_compatible = False
                else:
                    x_prime = T._transition_map[dyn_trans]
                    o_x_prime = T.observe(x_prime)
                    spec_trans = (q, o_x_prime)
                    specification_compatible = spec_trans in A._transition_map

                if dynamically_compatible and specification_compatible:
                    q_prime, trans_prob = A._get_next_state(q, o_x_prime)
                    q_final_prob = A._get_node_data(q, "final_probability")
                    q_prime_final_prob = A._get_node_data(q_prime, "final_probability")
                    o_x = T.observe(x)

                    (nodes, edges, _, _) = cls._add_product_edge(
                        T,
                        nodes,
                        edges,
                        x_src=x,
                        x_dest=x_prime,
                        q_src=q,
                        q_dest=q_prime,
                        q_src_final_prob=q_final_prob,
                        q_dest_final_prob=q_prime_final_prob,
                        observation_src=o_x,
                        observation_dest=o_x_prime,
                        sigma=sigma,
                        trans_prob=trans_prob,
                    )

                    prod_dest_state = (x_prime, q_prime)
                    if prod_dest_state not in visited:
                        visited.add(prod_dest_state)
                        search_queue.put(prod_dest_state)

        return cls._package_data(T, nodes, edges, init_prod_state)

    @classmethod
    def _get_product_state_label(
        cls, dynamical_system_state: Node, specification_state: Node
    ) -> Node:
        """
        Computes the combined product state label

        :param      dynamical_system_state:  The dynamical system state label
        :param      specification_state:     The specification state label

        :returns:   The product state label.
        """

        if type(dynamical_system_state) != str:
            dynamical_system_state = str(dynamical_system_state)

        if type(specification_state) != str:
            specification_state = str(specification_state)

        return dynamical_system_state + ", " + specification_state

    @classmethod
    def _breakdown_product_state(cls, product_state: Node) -> Tuple[Node, Node]:
        """
        Gets the dynamical system and specification states from a product state

        :param      product_state:  The product state label

        :returns:   dynamical system state label, specification state label
        """

        m = re.findall(r"(.+?)(?:,\s*|$)", product_state)

        x = m[0]
        q = m[1]

        return x, q

    @classmethod
    def _add_product_node(
        cls,
        dynamical_system: TransitionSystem,
        nodes: dict,
        x: Node,
        q: Node,
        q_final_prob: Probability,
        observation: Observation,
    ) -> Tuple[dict, Node]:
        """
        Adds a newly identified product state to the nodes dict w/ needed data

        :param      dynamical_system:  The dynamical system
        :param      nodes:             dict of nodes to build the product out
                                       of. must be in the format needed by
                                       networkx.add_nodes_from()
        :param      x:                 state label in the dynamical system
        :param      q:                 state label in the specification
        :param      q_final_prob:      the probability of terminating at q in
                                       the specification
        :param      observation:       The observation emitted by the dynamical
                                       system / product at the dynamical system
                                       state (x)

        :returns:   nodes dict populated with all of the given data, and the
                    label of the newly added product state
        """

        prod_state = cls._get_product_state_label(x, q)
        is_violating = q == SPEC_VIOLATING_STATE

        if prod_state not in nodes:
            prod_state_data = {
                "final_probability": q_final_prob,
                "trans_distribution": None,
                "is_violating": is_violating,
                "is_accepting": None,
                "observation": observation,
            }

            if "color" in dynamical_system.nodes[x]:
                color = dynamical_system.nodes[x]["color"]
                prod_state_data.update({"color": color})

            nodes[prod_state] = prod_state_data

        return nodes, prod_state

    @classmethod
    def _add_product_edge(
        cls,
        dynamical_system: TransitionSystem,
        nodes: dict,
        edges: dict,
        x_src: Node,
        x_dest: Node,
        q_src: Node,
        q_dest: Node,
        q_src_final_prob: Probability,
        q_dest_final_prob: Probability,
        observation_src: Observation,
        observation_dest: Observation,
        sigma: Symbol,
        trans_prob: Probability,
    ) -> Tuple[dict, dict, Node, Node]:
        """
        Adds a newly identified product edge to the nodes & edges dicts

        :param      dynamical_system:   The dynamical system
        :param      nodes:              dict of nodes to build the product out
                                        of. Must be in the format needed by
                                        networkx.add_nodes_from()
        :param      edges:              dict of edges to build the product out
                                        of. Must be in the format needed by
                                        networkx.add_edges_from()
        :param      x_src:              source product edge's dynamical system
                                        state
        :param      x_dest:             dest. product edge's dynamical system
                                        state
        :param      q_src:              source product edge's specification
                                        state
        :param      q_dest:             dest. product edge's specification
                                        state
        :param      q_src_final_prob:   the probability of terminating at q_src
                                        in the specification
        :param      q_dest_final_prob:  the probability of terminating at
                                        q_dest in the specification
        :param      observation_src:    The observation emitted by the
                                        dynamical system / product at the
                                        source dynamical system state (x_src)
        :param      observation_dest:   The observation emitted by the
                                        dynamical system / product at the
                                        dest. dynamical system state (x_dest)
        :param      sigma:              dynamical system control input symbol
                                        enabling the product edge
        :param      trans_prob:         The product edge's transition prob.

        :returns:   nodes dict populated w/ all the given data for src & dest
                    edges dict populated w/ all the given data,
                    the label of the newly added source product state,
                    the label of the newly added product state
        """

        nodes, prod_src = cls._add_product_node(
            dynamical_system, nodes, x_src, q_src, q_src_final_prob, observation_src
        )
        nodes, prod_dest = cls._add_product_node(
            dynamical_system, nodes, x_dest, q_dest, q_dest_final_prob, observation_dest
        )
        prod_edge_data = {"symbols": [sigma], "probabilities": [trans_prob]}
        prod_edge = {prod_dest: prod_edge_data}

        if prod_src in edges:
            if prod_dest in edges[prod_src]:
                existing_edge_data = edges[prod_src][prod_dest]

                existing_edge_data["symbols"].extend(prod_edge_data["symbols"])
                new_probs = prod_edge_data["probabilities"]
                existing_edge_data["probabilities"].extend(new_probs)

                edges[prod_src][prod_dest] = existing_edge_data
            else:
                edges[prod_src].update(prod_edge)
        else:
            edges[prod_src] = prod_edge

        return nodes, edges, prod_src, prod_dest

    @classmethod
    def _package_data(
        cls,
        dynamical_system: TransitionSystem,
        nodes: dict,
        edges: dict,
        init_prod_state: Node,
    ) -> dict:
        config_data = {}

        final_sym = dynamical_system.final_transition_sym
        config_data["final_transition_sym"] = final_sym
        empty_sym = dynamical_system.empty_transition_sym
        config_data["empty_transition_sym"] = empty_sym

        # can directly compute these from the graph data
        symbols = set()
        state_labels = set()
        observations = set()
        for state, edge in edges.items():
            for _, edge_data in edge.items():
                symbols.update(edge_data["symbols"])
                state_labels.add(state)

        for node in nodes.keys():
            observation = nodes[node]["observation"]
            observations.add(observation)

        alphabet_size = len(symbols)
        num_states = len(state_labels)
        num_obs = len(observations)

        config_data["alphabet_size"] = alphabet_size
        config_data["num_states"] = num_states
        config_data["num_obs"] = num_obs

        (symbol_display_map, nodes, edges) = Automaton._convert_states_edges(
            nodes, edges, final_sym, empty_sym, is_stochastic=IS_STOCHASTIC
        )
        config_data["nodes"] = nodes
        config_data["edges"] = edges
        config_data["start_state"] = init_prod_state
        config_data["symbol_display_map"] = symbol_display_map

        return config_data


class ProductBuilder(Builder):
    """
    Implements the generic automaton builder class for Product objects
    """

    def __init__(self) -> "ProductBuilder":
        """
        Constructs a new instance of the ProductBuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(
        self,
        graph_data: Tuple[TransitionSystem, PDFA],
        graph_data_format: str = "existing_objects",
        **kwargs: dict,
    ) -> Product:
        """
        Returns an initialized Product instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The graph configuration file name
        :param      graph_data_format:  The graph data file format.
                                        {'existing_objects'}
        :param      kwargs:             The keywords arguments to the specific
                                        constructors

        :returns:   instance of an initialized Product object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
        """

        if graph_data_format == "existing_objects":
            sys = graph_data[0]
            spec = graph_data[1]
            self._instance = self._from_automata(
                dynamical_system=sys, specification=spec, **kwargs
            )
        else:
            msg = (
                "graph_data_format ({}) must be one of: "
                + '"existing_objects"'.format()
            )
            raise ValueError(msg)

        return self._instance

    def _from_automata(
        self,
        dynamical_system: TransitionSystem,
        specification: PDFA,
        normalize_trans_probabilities: bool = False,
        show_steps: bool = False,
    ) -> Product:
        """
        Returns an instance of a Product Automaton from existing automata

        :param      dynamical_system:               The dynamical system
                                                    automaton instance
        :param      specification:                  The specification automaton
                                                    instance
        :param      normalize_trans_probabilities:  whether to renormalize the
                                                    edge probabilities such
                                                    that each states has a well
                                                    defined transition
                                                    probability distribution.
                                                    We typically DONT want to
                                                    modify the probabilities
                                                    of the product algorithm,
                                                    except if we would like
                                                    to be able to easily sample
                                                    from the automaton.
        :param      show_steps:                     draw intermediate steps in
                                                    the product creation

        :returns:   instance of an initialized Product automaton object
        """

        # don't want to destroy the automaton when we pre-process them
        internal_dyn_sys = copy.deepcopy(dynamical_system)

        augmented_dyn_sys = Product._augment_initial_state(
            internal_dyn_sys, specification
        )
        if show_steps:
            augmented_dyn_sys.draw("augment_initial_state")

        config_data = Product._compute_product(augmented_dyn_sys, specification)

        config_data["is_normalized"] = normalize_trans_probabilities

        # saving these so we can just return initialized instances if the
        # underlying data has not changed
        self.nodes = config_data["nodes"]
        self.edges = config_data["edges"]

        if not self.edges:
            msg = "no compatible edges were found, so the product is empty"
            warnings.warn(msg, RuntimeWarning)
            self._instance = None
        else:
            self._instance = Product(**config_data)

        return self._instance
