import collections
import os
from typing import Tuple

from bidict import bidict

from specless.factory.builder import Builder
from specless.typing import ActionsEnum, EnvAct, EnvActs
from specless.wrapper.tswrapper import StepData, TransitionSystemWrapper

from .base import (
    DEFAULT_EMPTY_TRANS_SYMBOL,
    DEFAULT_FINAL_TRANS_SYMBOL,
    Automaton,
    Node,
    Nodes,
    NXEdgeList,
    NXNodeList,
    Observation,
    Symbol,
    Symbols,
)

# define these type defs for method annotation type hints
TS_Trans_Data = Tuple[Node, Observation]


class TransitionSystem(Automaton):
    """
    A representation of a transition system automaton (a.k.a. moore machine)

    :param      nodes:                 node list as expected by
                                       networkx.add_nodes_from() (node
                                       label, node attribute dict)
    :param      edges:                 edge list as expected by
                                       networkx.add_edges_from() (src node
                                       label, dest node label, edge
                                       attribute dict)
    :param      symbol_display_map:    bidirectional mapping of
                                       hashable symbols, to a unique
                                       integer index in the symbol map.
                                       Needed to translate between the
                                       indices in the transition
                                       distribution and the hashable
                                       representation which is
                                       meaningful to the user
    :param      alphabet_size:         number of symbols in system alphabet
    :param      num_states:            number of states in automaton state
                                       space
    :param      num_obs:               number of observation symbols
    :param      start_state:           unique start state string label of
                                       system
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
        num_obs: int,
        final_transition_sym: {Symbol, None} = DEFAULT_FINAL_TRANS_SYMBOL,
        empty_transition_sym: {Symbol, None} = DEFAULT_EMPTY_TRANS_SYMBOL,
    ) -> None:
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
            num_obs=num_obs,
            final_transition_sym=final_transition_sym,
            empty_transition_sym=empty_transition_sym,
            state_observation_key="observation",
            can_have_accepting_nodes=True,
            edge_weight_key=None,
        )

    def transition(
        self, curr_state: Node, input_symbol: Symbol, **get_next_state_kwargs: dict
    ) -> TS_Trans_Data:
        """
        transitions the TS given the current TS state and an input symbol, then
        outputs the state observation

        :param      curr_state:             The current TS state
        :param      input_symbol:           The input TS symbol
        :param      get_next_state_kwargs:  Any additional inputs to
                                            _get_next_state

        :returns:   the next TS state, and the obs
        """

        next_state, _ = self._get_next_state(
            curr_state, input_symbol, **get_next_state_kwargs
        )
        observation = self.observe(next_state)

        return next_state, observation

    def observe(self, curr_state: Node) -> Observation:
        """
        Returns the given state's observation symbol

        :param      curr_state:  The current TS state

        :returns:   observation symbol emitted at curr_state
        """

        return self._get_node_data(curr_state, "observation")

    def run(
        self, word: {Symbol, Symbols}, **get_next_state_kwargs: dict
    ) -> Tuple[Symbols, Nodes]:
        """
        processes a input word and produces a output word & state sequence

        :param      word:                   The word to process
        :param      get_next_state_kwargs:  Any additional inputs to
                                            _get_next_state

        :returns:   output word (list of symbols), list of states visited

        :raises     ValueError:             Catches and re-raises exceptions
                                            from invalid symbol use
        """

        # need to do type-checking / polymorphism handling here
        if isinstance(word, str) or not isinstance(word, collections.Iterable):
            word = [word]

        curr_state = self.start_state
        output_word = [self.observe(curr_state)]
        state_sequence = [curr_state]

        for symbol in word:
            try:
                next_state, observation = self.transition(
                    curr_state, symbol, **get_next_state_kwargs
                )
            except ValueError as e:
                msg = (
                    "Invalid symbol encountered processesing "
                    + f"word: {word}.\ncurrent output word: {output_word}"
                    + f" \ncurrent state sequence: {state_sequence}"
                )
                raise ValueError(msg) from e

            output_word.append(observation)
            state_sequence.append(next_state)

            curr_state = next_state

        return output_word, state_sequence

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        TS doesn't accept anything, so this just passes
        """
        pass


class MinigridTransitionSystem(TransitionSystem):
    """
    A class representing both a transition system and a Minigrid environment.


    The Minigrid gym environment is kept in sync with the transition system

    NOTE: changes to `_env` are NOT
    automatically recognized by the TS, so only use methods of this class to
    change attributes of `_env` unless you know what you're doing.

    :param      nodes:                 node list as expected by
                                       networkx.add_nodes_from() (node
                                       label, node attribute dict)
    :param      edges:                 edge list as expected by
                                       networkx.add_edges_from() (src node
                                       label, dest node label, edge
                                       attribute dict)
    :param      symbol_display_map:    bidirectional mapping of
                                       hashable symbols, to a unique
                                       integer index in the symbol map.
                                       Needed to translate between the
                                       indices in the transition
                                       distribution and the hashable
                                       representation which is
                                       meaningful to the user
    :param      alphabet_size:         number of symbols in system alphabet
    :param      num_states:            number of states in automaton state
                                       space
    :param      num_obs:               number of observation symbols
    :param      start_state:           unique start state string label of
                                       system
    :param      final_transition_sym:  representation of the termination
                                       symbol. If not given, will default
                                       to base class default.
    :param      empty_transition_sym:  representation of the empty symbol
                                       (a.k.a. lambda). If not given, will
                                       default to base class default.
    :param      env:                   Minigrid gym env represented by this TS
    """

    current_state: Node
    """the current state in the transition system. Kept in sync with env"""

    def __init__(self, **kwargs):
        self.env = kwargs["env"]

        # normal TS don't have an 'env'
        kwargs.pop("env", None)
        super().__init__(**kwargs)

        self.actions = self.env.actions
        """actions available in the gym env. Can be fed into the TS or env"""

        self.reset()

    # TODO: Remove _get_state_from_str
    def reset(self) -> None:
        """
        Resets both the transition system's state and the Minigrid env itself.

        :param      new_monitor_file:  whether to create a new monitor file
        """

        self.env.reset()
        self.current_state = self.start_state

    # TODO: Remove OBS!!!
    def step(self, action: ActionsEnum) -> StepData:
        obs, reward, done, _ = self.env.state_only_obs_step(action)

        (
            dest_state,
            _,
            terminated,
            truncated,
            _,
        ) = self.env.make_transition(self.current_state, action)

        # update states
        self.current_state = dest_state

        return obs, reward, done, _

    def transition(
        self,
        curr_state: Node,
        input_symbol: {Symbol, EnvAct},
        **get_next_state_kwargs: dict,
    ) -> TS_Trans_Data:
        """
        transitions the TS given the current TS state and an input symbol, then
        outputs the state observation.

        Note: Accepts both a TS symbol or one of self.actions.

        :param      curr_state:             The current TS state
        :param      input_symbol:           The input TS symbol
        :param      get_next_state_kwargs:  Any additional inputs to
                                            _get_next_state

        :returns:   the next TS state, and the obs
        """

        return super().transition(curr_state, input_symbol, **get_next_state_kwargs)

    def run(
        self, word: {EnvAct, EnvActs, Symbol, Symbols}, **get_next_state_kwargs: dict
    ) -> Tuple[Symbols, Nodes]:
        """processes a input word and produces a output word & state sequence"""

        self.reset()

        # need to do type-checking / polymorphism handling here
        if isinstance(word, str) or not isinstance(word, collections.Iterable):
            word = [word]

        output_word, state_sequence = super().run(word, **get_next_state_kwargs)

        return output_word, state_sequence

    def _get_next_state(
        self, curr_state: Node, symbol: {Symbol, EnvAct}
    ) -> Tuple[Node, float]:
        """
        Gets the next state given the current state and the "input" symbol.

        computes this using the underlying environment's step() function.
        Note: Accepts both a TS symbol or one of self.actions.

        """

        if isinstance(symbol, self.actions):
            symbol = self.env.ACTION_ENUM_TO_STR[symbol]

        (possible_symbols, _) = self._get_trans_probabilities(curr_state)

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

        symbol_probability = None

        # Remove _get_state_from_str, _make_transition, _get_state_str
        # We need to convert
        # curr_state[str] + symbol[str] => next_state[str]
        # to
        # curr_node[Tuple] + action[ActionsEnum] => next_node[Tuple]
        action = self.env.ACTION_STR_TO_ENUM[symbol]

        (
            dest_state,
            _,
            terminated,
            truncated,
            _,
        ) = self.make_transition(curr_state, action)
        # dest_node = self._get_node_from_state(dest_state)

        self.current_state = dest_state

        # need to make sure that the environment and the internal, TS
        # transition map are in sync
        dest_states = self._transition_map[(curr_state, symbol)]
        if dest_state != dest_states:
            msg = (
                f"At the current_state ({curr_state}) under given symbol "
                + f"({symbol}, the internal Minigrid env"
                "s env.step() "
                + f"returned a next state ({dest_state}) that was "
                + "different than the next state "
                + f"({dest_states}) in the transition system."
            )
            raise ValueError(msg)

        return dest_state, symbol_probability


class TSBuilder(Builder):
    """
    Implements the generic automaton builder class for TransitionSystem objects
    """

    def __init__(self) -> None:
        """
        Constructs a new instance of the TSBuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(
        self,
        graph_data: {str, TransitionSystemWrapper},
        graph_data_format: str = "yaml",
    ) -> TransitionSystem:
        """
        Returns an initialized TransitionSystem instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The graph configuration data
                                        {str, Minigrid environment wrapper}
        :param      graph_data_format:  The graph data file format.
                                        {'yaml', 'minigrid'}

        :returns:   instance of an initialized TransitionSystem object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
        """

        if graph_data_format == "yaml":
            config_data = self._from_yaml(graph_data)
            TS_Type = TransitionSystem
        elif graph_data_format == "minigrid":
            config_data: TransitionSystem = self._from_minigrid(graph_data)
            TS_Type = MinigridTransitionSystem
        else:
            msg = (
                'graph_data_format ({}) must be one of: "yaml", '
                + '"minigrid"'.format()
            )
            raise ValueError(msg)

        nodes_have_changed = self.nodes != config_data["nodes"]
        edges_have_changed = self.edges != config_data["edges"]
        no_instance_loaded_yet = self._instance is None

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:
            # nodes and edge_list must be in the format needed by:
            #   - networkx.add_nodes_from()
            #   - networkx.add_edges_from()
            if "final_transition_sym" not in config_data:
                final_transition_sym = DEFAULT_FINAL_TRANS_SYMBOL
            else:
                final_transition_sym = config_data["final_transition_sym"]

            if "empty_transition_sym" not in config_data:
                empty_transition_sym = DEFAULT_EMPTY_TRANS_SYMBOL
            else:
                empty_transition_sym = config_data["empty_transition_sym"]

            (symbol_display_map, states, edges) = Automaton._convert_states_edges(
                config_data["nodes"],
                config_data["edges"],
                final_transition_sym,
                empty_transition_sym,
                is_stochastic=False,
            )
            config_data["symbol_display_map"] = symbol_display_map

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = states
            self.edges = edges
            config_data["nodes"] = self.nodes
            config_data["edges"] = self.edges

            self._instance = TS_Type(**config_data)

        return self._instance

    def _from_minigrid(
        self, minigrid_environment: TransitionSystemWrapper
    ) -> TransitionSystem:
        config_data = minigrid_environment.extract_transition_system()
        config_data["env"] = minigrid_environment

        return config_data

    def _from_yaml(self, graph_data: str) -> dict:
        _, file_extension = os.path.splitext(graph_data)

        allowed_exts = [".yaml", ".yml"]
        if file_extension in allowed_exts:
            config_data = self.load_YAML_config_data(graph_data)
        else:
            msg = "graph_data ({}) is not a ({}) file"
            raise ValueError(msg.format(graph_data, allowed_exts))

        return config_data
