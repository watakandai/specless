# 3rd-party packages
import re
from typing import Tuple

import networkx as nx

# import pygraphviz
from bidict import bidict
from networkx.drawing import nx_agraph
from networkx.drawing.nx_pydot import read_dot

# local packages
from specless.factory.builder import Builder

from .base import DEFAULT_EMPTY_TRANS_SYMBOL, DEFAULT_FINAL_TRANS_SYMBOL, Automaton
from .types import Node, NXEdgeList, NXNodeList, Symbol


class FDFA(Automaton):
    """
    This class describes a frequency deterministic finite automaton (fdfa).

    built on networkx, so inherits node and edge data structure definitions

    Node Attributes
    -----------------
        - final_frequency: final state frequency for each node.
                           Number of times that a trace ended in that state.
        - in_frequency:    in "flow" of state frequency for each node
                           total times that state was visited with incoming
                           transitions.
        - out_frequency:   out "flow" of state frequency for each node
                           total times that state was visited with outgoing
                           transitions.
        - trans_distribution: None, just there for consistency with PDFA
        - is_accepting: None, just there for consistency with PDFA

    Edge Properties
    -----------------
        - symbol: the symbol value emitted when the edge is traversed
        - frequency: the number of times the edge was traversed

    :param      nodes:                 node list as expected by
                                       networkx.add_nodes_from() list of
                                       tuples: (node label, node, attribute
                                       dict)
    :param      edges:                 edge list as expected by
                                       networkx.add_edges_from() list of
                                       tuples: (src node label, dest node
                                       label, edge attribute dict)
    :param      symbol_display_map:    bidirectional mapping of hashable
                                       symbols, to a unique integer index
                                       in the symbol map. Needed to
                                       translate between the indices in the
                                       transition distribution and the
                                       hashable representation which is
                                       meaningful to the user
    :param      alphabet_size:         number of symbols in fdfa alphabet
    :param      num_states:            number of states in automaton state
                                       space
    :param      start_state:           unique start state string label of
                                       fdfa
    :param      final_transition_sym:  representation of the termination
                                       symbol. If not given, will default
                                       to base class default.
    :param      empty_transition_sym:  representation of the empty symbol
                                       (a.k.a. lambda). If not given, will
                                       default to base class default.
    """

    def __init__(
        self,
        nodes: NXNodeList,
        edges: NXEdgeList,
        symbol_display_map: bidict,
        alphabet_size: int,
        num_states: int,
        start_state: Node,
        final_transition_sym: {Symbol, None} = None,
        empty_transition_sym: {Symbol, None} = None,
    ) -> "FDFA":
        # need to start with a fully initialized automaton
        super().__init__(
            nodes,
            edges,
            symbol_display_map,
            alphabet_size,
            num_states,
            start_state,
            smooth_transitions=False,
            is_stochastic=False,
            is_sampleable=True,
            num_obs=None,
            final_transition_sym=final_transition_sym,
            initial_weight_key="initial_frequency",
            final_weight_key="final_frequency",
            can_have_accepting_nodes=False,
            edge_weight_key="frequency",
        )

    def to_pdfa_data(self) -> Tuple[NXNodeList, NXEdgeList]:
        """
        convert self nodes and edges to pdfa nodes and edges

        :returns:   nodes, edges lists with all data initialized for creation
                    of pdfa from networkx.add_nodes_from() and
                    networkx.add_edges_from()
        :rtype:     list of tuples: (node label, node, attribute dict),
                    list of tuples: (src node label, dest node label,
                                     edge attribute dict)
        """

        fdfa_nodes = self.nodes(data=True)
        pdfa_nodes = []
        pdfa_edges = []

        # converting final state frequencies to final state probabilities
        for curr_node, curr_node_data in fdfa_nodes:
            # the final probability is just how often the execution ends at the
            # curr_node divided by the all of sum of frequencies over all
            # possible transitions from that node
            final_freq = self._get_node_data(curr_node, "final_frequency")
            out_freq = self._get_node_data(curr_node, "out_frequency")
            number_of_choices = final_freq + out_freq
            new_final_probability = final_freq / number_of_choices

            new_node_data = {
                "final_probability": new_final_probability,
                "trans_distribution": None,
                "is_accepting": None,
            }
            pdfa_nodes.append((curr_node, new_node_data))

            # converting transition frequencies to transition probabilities
            #
            # the edge transition probability is the edge's frequency divided
            # by the the number of time you either ended or transitioned out
            # of the that node
            for node_post in self.successors(curr_node):
                curr_edges_out = self.get_edge_data(curr_node, node_post)

                for _, curr_out_edge_data in curr_edges_out.items():
                    edge_freq = curr_out_edge_data["frequency"]
                    symbol = curr_out_edge_data["symbol"]
                    trans_probability = edge_freq / number_of_choices
                    new_edge_data = {"symbol": symbol, "probability": trans_probability}

                    new_edge = (curr_node, node_post, new_edge_data)

                    pdfa_edges.append(new_edge)

        return pdfa_nodes, pdfa_edges

    @classmethod
    def load_flexfringe_data(
        cls: "FDFA",
        graph: nx.MultiDiGraph,
        number_input_symbols: int,
        final_transition_sym: Symbol,
        empty_transition_sym: Symbol,
    ) -> dict:
        """
        reads in graph configuration data from a flexfringe dot file

        :param      cls:                   The "class instance" this method
                                           belongs to (not object instance)
        :param      graph:                 The nx graph with the flexfringe
                                           fdfa model loaded in
        :param      number_input_symbols:  The number of input symbols to the
                                           FDFA needed to compute the correct
                                           frequency flows in the case of
                                           cycles.
        :param      final_transition_sym:  representation of the empty string /
                                           symbol (a.k.a. lambda)
        :param      empty_transition_sym:  The empty transition symbol

        :returns:   configuration data dictionary for the fdfa
        :rtype:     dictionary
        """

        # flexfringe always labels the root as 0
        root_node_label = "0"

        ff_nodes = graph.nodes(data=True)
        ff_edges = graph.edges(data=True)

        (nodes, node_ID_to_node_label) = cls.convert_flexfringe_nodes(
            ff_nodes, number_input_symbols, root_node_label
        )
        (symbol_display_map, edges, symbols) = cls.convert_flexfringe_edges(
            ff_edges, final_transition_sym, empty_transition_sym, node_ID_to_node_label
        )
        config_data = {
            "nodes": nodes,
            "edges": edges,
            "symbol_display_map": symbol_display_map,
            "alphabet_size": len(symbols),
            "num_states": len(nodes),
            "start_state": node_ID_to_node_label[root_node_label],
        }

        return config_data

    @staticmethod
    def convert_flexfringe_nodes(
        flexfringe_nodes: dict, number_input_symbols: int, root_node_label: Node
    ) -> (NXNodeList, dict):
        """
        converts node data from a flexfringe (FF) dot file into the internal
        node format needed by networkx.add_nodes_from()

        :param      flexfringe_nodes:      The flexfringe node list mapping
                                           node labels to node attributes
        :param      number_input_symbols:  The number of input symbols to the
                                           FDFA needed to compute the correct
                                           frequency flows in the case of
                                           cycles.
        :param      root_node_label:       The root node's label

        :returns:   node list as expected by networkx.add_nodes_from(), a dict
                    mapping FF node IDs to FF state labels

        :raises     ValueError:            can't read in "blue" flexfringe
                                           nodes, as they are theoretically
                                           undefined for this class right now
        """

        nodes = {}
        node_ID_to_node_label = {}

        for node_ID, node_data in flexfringe_nodes:
            if "label" not in node_data:
                continue

            state_label = re.findall(r"\d+", node_data["label"])

            # we can't add blue nodes to our graph
            if "style" in node_data:
                if "dotted" in node_data["style"]:
                    err = (
                        "node = {} from flexfringe is blue,"
                        " reading in blue states is not"
                        " currently supported"
                    ).format(node_data)
                    raise ValueError(err)

            # until we read in all of the nodes, have to wait to find the
            # initial state
            initial_frequency = 0

            new_node_label = "q" + str(node_ID)
            new_node_data = {
                "final_frequency": 0,
                "initial_frequency": initial_frequency,
                "trans_distribution": None,
                "isAccepting": None,
            }

            nodes[new_node_label] = new_node_data
            node_ID_to_node_label[node_ID] = new_node_label

        # need to give all of the frequency flow to the start node
        start_state = node_ID_to_node_label[root_node_label]
        nodes[start_state]["initial_frequency"] = number_input_symbols

        # best convention is to convert dict_items to a list, even though both
        # are iterable
        converted_nodes = list(nodes.items())

        return converted_nodes, node_ID_to_node_label

    @staticmethod
    def convert_flexfringe_edges(
        flexfringeEdges: NXEdgeList,
        final_transition_sym: Symbol,
        empty_transition_sym: Symbol,
        node_ID_to_node_label: dict,
    ) -> (bidict, NXEdgeList, set):
        """
        converts edges read in from flexfringe (FF) dot files into the internal
        edge format needed by networkx.add_edges_from()

        :param      flexfringeEdges:        The flexfringe edge list mapping
                                            edges labels to edge attributes
        :param      final_transition_sym:   representation of the termination
                                            symbol
        :param      empty_transition_sym:   representation of the empty symbol
                                            (a.k.a. lambda).
        :param      node_ID_to_node_label:  mapping from FF node ID to FF node
                                            label

        :returns:   symbol_display_map - bidirectional mapping of hashable
                    symbols, to a unique integer index in the symbol map,
                    edge list as expected by networkx.add_edges_from(),
                    set of observed symbols
        """

        edges = []
        seen_symbols = []

        # add these symbols first, so we can then later ensure they have the
        # last two indices
        seen_symbols.append(empty_transition_sym)
        seen_symbols.append(final_transition_sym)
        symbol_display_map = bidict({})

        for src_FF_node_ID, dest_FF_node_ID, edge_data in flexfringeEdges:
            new_edge_data = {}

            if "label" not in edge_data:
                continue

            transitionData = re.findall(r"([\w]*):(\d+)", edge_data["label"])
            symbols, frequencies = zip(*transitionData)

            for symbol, frequency in transitionData:
                # need to store new symbols in a map for display
                if symbol not in seen_symbols:
                    seen_symbols.append(symbol)

                new_edge_data = {"symbol": str(symbol), "frequency": int(frequency)}

                src_FF_node_label = node_ID_to_node_label[src_FF_node_ID]
                dest_FF_node_label = node_ID_to_node_label[dest_FF_node_ID]
                new_edge = (src_FF_node_label, dest_FF_node_label, new_edge_data)

                edges.append(new_edge)

        # ensure that the empty and final symbols always have the last indices
        # in the display map for use in computations excluding those symbols
        symbol_display_map = bidict({})
        for new_sym_idx, symbol in enumerate(reversed(seen_symbols)):
            symbol_display_map[symbol] = new_sym_idx

        return symbol_display_map, edges, set(seen_symbols)

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        FDFA doesn't accept anything, so this just passes
        """
        pass

    def _compute_node_data_properties(self, curr_node: Node, **node_data_args) -> None:
        """
        Sets all state frequencies for each node in an initialized FDFA

        requires self.nodes and self.edges to be properly loaded into nx data
        structures

        :warning this overrides the base _compute_node_data_properties method
                 in the Automaton

        :param      curr_node:   The node to set properties for

        :returns:   None

        :raises     ValueError:  checks if the final frequency is less than 0,
                                 indicating something wrong with the edge
                                 frequency data
        """

        number_trans_in = self._compute_node_flow(curr_node, flow_type="in")
        number_trans_out = self._compute_node_flow(curr_node, flow_type="out")

        init_trans = self._get_node_data(curr_node, "initial_frequency")

        # the final frequency is simply the number of times that you
        # transitioned into a state and then did not leave it.
        #
        # inflow and outflow must not include self transitions, as it self
        # transitions are not true flow
        curr_node_final_freq = init_trans + number_trans_in - number_trans_out

        # no node is allowed to "create" transitions
        if curr_node_final_freq < 0:
            err = (
                "current node ({}) final frequency ({}) should "
                + "not be less than 0. This means there were more "
                + "outgoing transitions ({}) than incoming "
                + "transitions ({})."
            )
            raise ValueError(
                err.format(
                    curr_node, curr_node_final_freq, number_trans_out, number_trans_in
                )
            )

        self._set_node_data(curr_node, "final_frequency", curr_node_final_freq)
        self._set_node_data(curr_node, "in_frequency", number_trans_in)
        self._set_node_data(curr_node, "out_frequency", number_trans_out)

        # need to compute the transition map
        edge_data = self.edges([curr_node], data=True)
        edge_dests = [edge[1] for edge in edge_data]

        original_edge_symbols = [edge[2]["symbol"] for edge in edge_data]
        edge_symbols = [
            self._symbol_display_map[symbol] for symbol in original_edge_symbols
        ]
        self._set_trans_map(curr_node, edge_symbols, edge_dests)

    def _compute_node_flow(self, curr_node: Node, flow_type: str) -> (int, int):
        """
        Calculates frequency (in/out)flow at the current node

        :param      curr_node:   The node to compute the flow at
        :param      flow_type:   The flow type {'in', 'out'}

        :returns:   The node's (in/out)flow, the node's self-transition flow

        :raises     ValueError:  checks if flow_type is an supported setting
        """

        allowed_flow_types = ["in", "out"]
        if flow_type not in allowed_flow_types:
            msg = (
                "selected flow_type ({}) not one of " "allowed flow_types: {}"
            ).format(flow_type, allowed_flow_types)
            raise ValueError(msg)

        if flow_type == "in":
            nodes = self.predecessors(curr_node)
        elif flow_type == "out":
            nodes = self.successors(curr_node)

        number_trans = 0
        for node in nodes:
            if flow_type == "in":
                curr_edges = self.get_edge_data(node, curr_node)
            elif flow_type == "out":
                curr_edges = self.get_edge_data(curr_node, node)

            for _, curr_out_edge_data in curr_edges.items():
                frequency = curr_out_edge_data["frequency"]
                number_trans += frequency

        return number_trans


class FDFABuilder(Builder):
    """
    Implements the generic automaton builder class for FDFA objects
    """

    def __init__(self) -> "FDFABuilder":
        """
        Constructs a new instance of the FDFABuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(
        self,
        graph_data: str,
        number_input_symbols: int = None,
        graph_data_format: str = "dot_string",
    ) -> FDFA:
        """
        Returns an initialized FDFA instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:            The string containing graph data.
                                           Could be a filename or just the raw
                                           data
        :param      number_input_symbols:  The number of input symbols to the
                                           FDFA needed to compute the correct
                                           frequency flows in the case of
                                           cycles.
                                           Only really optional when using a
                                           graph_data_format
                                           that already has this information.
        :param      graph_data_format:     The graph data file format.
                                           {'dot_file', 'dot_string',
                                            'learning_interface'}

        :returns:   instance of an initialized FDFA object

        :raises     ValueError:            checks if graph_data and
                                           graph_data_format have a compatible
                                           data loader.
        :raises     ValueError:            checks, based on graph_data_format,
                                           whether it is legal to not specify
                                           the number_input_symbols.
        """

        has_number_input_symbols = number_input_symbols is not None
        if graph_data_format == "dot_string":
            graph = nx_agraph.from_agraph(pygraphviz.AGraph(string=graph_data))
        elif graph_data_format == "dot_file":
            graph = read_dot(graph_data)
        elif graph_data_format == "learning_interface":
            learning_interface = graph_data
            graph = read_dot(learning_interface.learned_model_filepath)
            number_input_symbols = learning_interface.num_training_examples
            has_number_input_symbols = True
        else:
            msg = (
                'graph_data_format ({}) must be one of: "dot_file", '
                + '"dot_string"'.format()
            )
            raise ValueError(msg)

        if not has_number_input_symbols:
            msg = "must provide the number_input_symbols to load a FDFA"
            raise ValueError(msg)

        # these are not things that are a part of flexfringe's automaton
        # data model, so give them default values
        final_transition_sym = DEFAULT_FINAL_TRANS_SYMBOL
        empty_transition_sym = DEFAULT_EMPTY_TRANS_SYMBOL
        config_data = FDFA.load_flexfringe_data(
            graph, number_input_symbols, final_transition_sym, empty_transition_sym
        )
        config_data["final_transition_sym"] = final_transition_sym
        config_data["empty_transition_sym"] = empty_transition_sym

        nodes_have_changed = self.nodes != config_data["nodes"]
        edges_have_changed = self.edges != config_data["edges"]
        no_instance_loaded_yet = self._instance is None

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:
            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = config_data["nodes"]
            self.edges = config_data["edges"]

            self._instance = FDFA(**config_data)

        return self._instance
