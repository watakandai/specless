from __future__ import annotations

# 3rd-party packages
import os
import queue
import warnings
from typing import Callable, List, Tuple

from bidict import bidict

# local packages
from specless.factory.builder import Builder

from .base import SMOOTHING_AMOUNT, Automaton
from .dfa import SafetyDFA
from .fdfa import FDFA
from .types import (
    Node,
    NXEdgeList,
    NXNodeList,
    Probabilities,
    Probability,
    Symbol,
    Symbols,
)
from .utils import logx, xlogx, xlogy, ylogx

IS_STOCHASTIC = True
SPEC_VIOLATING_STATE = "q_v"


def check_predict_method(prediction_function: Callable):
    """
    decorator to check an enumerated typestring for prediction method.
    pred_method:  The pred_method string to check. one of: {'sample',
    'max_prob'}

    :type       prediction_function:  function handle to check. Must have
                                      keyword argument: 'pred_method'
    :param      prediction_function:  the function to decorate

    :raises     ValueError: raises if:
                                - pred_method is not a keyword argument
                                - pred_method is not one of allowed methods
    """

    def checker(*args, **kwargs):
        # checking if the decorator has been applied to an appropriate function
        print(args, kwargs)
        if "pred_method" not in kwargs:
            f_name = prediction_function.__name__
            msg = f'"pred_method" is not a kwarg of {f_name}'
            raise ValueError(msg)

        pred_method = kwargs["pred_method"]

        # checking for the enumerated types
        allowed_methods = ["max_prob", "sample"]

        if pred_method not in allowed_methods:
            msg = (
                f'pred_method: "{pred_method}" must be one of allowed '
                + f"methods: {allowed_methods}"
            )
            raise ValueError(msg)

        return prediction_function(*args, **kwargs)

    return checker


class PDFA(Automaton):
    """
    This class describes a probabilistic deterministic finite automaton (pdfa).

    built on networkx, so inherits node and edge data structure definitions

    inherits some of its api from the NLTK LM API

    Node Attributes
    -----------------
        - final_probability: final state probability for the node
        - trans_distribution: a sampled-able function to select the next state
          and emitted symbol
        - is_accepting: a boolean flag determining whether the pdfa considers
          the node accepting

    Edge Properties
    -----------------
        - symbol: the symbol value emitted when the edge is traversed
        - probability: the probability of selecting this edge for traversal

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
    :param      alphabet_size:         number of symbols in pdfa alphabet
    :param      num_states:            number of states in automaton state
                                       space
    :param      start_state:           unique start state string label of
                                       pdfa
    :param      smooth_transitions:    whether to smooth the symbol
                                       transitions distributions
    :param      smoothing_amount:      probability mass to re-assign to
                                       unseen symbols at each node
    :param      final_transition_sym:  representation of the termination
                                       symbol. If not given, will default
                                       to base class default.
    :param      empty_transition_sym:  representation of the empty symbol
                                       (a.k.a. lambda). If not given, will
                                       default to base class default.
    :param      beta:                  the final state probability needed
                                       for a state to accept.
    :param      merge_sinks:           whether to combine all states
                                       together that have no outgoing
                                       edges
    """

    def __init__(
        self,
        nodes: NXNodeList,
        edges: NXEdgeList,
        symbol_display_map: bidict,
        alphabet_size: int,
        num_states: int,
        start_state: Node,
        smooth_transitions: bool,
        smoothing_amount: float = SMOOTHING_AMOUNT,
        final_transition_sym: {Symbol, None} = None,
        empty_transition_sym: {Symbol, None} = None,
        beta: float = 0.95,
        merge_sinks: bool = False,
        is_normalized: bool = False,
    ) -> "PDFA":
        self._beta = beta
        """the final state probability needed for a state to accept"""

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
            smooth_transitions=smooth_transitions,
            is_stochastic=True,
            is_sampleable=is_sampleable,
            is_normalized=is_normalized,
            num_obs=None,
            final_transition_sym=final_transition_sym,
            empty_transition_sym=empty_transition_sym,
            final_weight_key="final_probability",
            can_have_accepting_nodes=True,
            edge_weight_key="probability",
            smoothing_amount=smoothing_amount,
            merge_sinks=merge_sinks,
        )

    def predict(self, symbols: Symbols, pred_method: str = "max_prob") -> Symbol:
        """
        predicts the next symbol conditioned on the given previous symbols

        :param      symbols:      The previously observed emitted symbols
        :param      pred_method:  The method used to choose the next state. see
                                  _choose_next_state for details on how each
                                  pred_method is implemented.
                                  {'sample', 'max_prob'}

        :returns:   the most probable next symbol in the sequence
        """

        # simulating the state trajectory under the given sequence
        state = self.start_state

        for symbol in symbols:
            state, _ = self._get_next_state(state, symbol)

        # now making the next state prediction based on the "causal" model
        # state induced by the deterministic sequence governed by the
        # observed symbols
        _, next_symbol, _ = self._choose_next_state(state)

        return next_symbol

    def score(self, trace: Symbols) -> float:
        """
        Calculates the given trace's probability in the language of the PDFA.

        PDFA is a language model (LM) in this case:
            ==> score = P_{PDFA LM}(trace)

        :param      trace:  A trace

        :returns:           A trace probability
        """

        curr_state = self.start_state
        trace_prob = 1.0
        n_symbol = len(trace)

        for symbol in trace:
            try:
                (next_state, trans_probability) = self._get_next_state(
                    curr_state, symbol
                )

            except ValueError as e:
                warnings.warn(str(e))
                return 0.0

            trace_prob *= trans_probability
            # print(curr_state, '-->', symbol, '-->', trans_probability, ', \tTotal: ', trace_prob)
            curr_state = next_state

        return trace_prob

    def scores(self, traces: List[Symbols]) -> List[float]:
        """
        Calculates trace probabilities

        :param      trace:  A list of traces

        :returns:   The trace probability.
        """
        return [self.score(trace) for trace in traces]

    def logscore(self, trace: Symbols, base: float = 2.0) -> float:
        """
        Computes the log of the score (sequence probability) of the given trace
        in the language of the PDFA

        :param      trace:  The sequence of symbols to compute the log score of
        :param      base:   The log base. Commonly set to 2 in classic
                            information theory literature

        :returns:   log of the probability - NOT log odds
        """

        score = self.score(trace)

        return logx(score, base)

    def logscores(self, traces: List[Symbols], **kwargs) -> List[float]:
        """
        Computes traces log probabilities

        :param      traces:     A list of traces
        :param      base:       The log base.

        :returns:               log of the probability
        """
        return [self.logscore(trace, **kwargs) for trace in traces]

    def cross_entropy_approx(self, trace: Symbols, base: float = 2.0) -> float:
        """
        computes approximate cross-entropy of the given trace in the language
        of the PDFA

        Here, we are using the Shannon-McMillian-Breiman theorem to define
        the cross-entropy of the trace, given that we sampled the trace from
        the actual target distribution and we are evaluating it in the PDFA LM.
        Then, as a PDFA is a stationary ergodic stochastic process model, we
        can calculate the cross-entropy as (eq. 3.49 from SLP ch3):

            trace ~ target
            H(target, model) = lim {(- 1 / n) * log(P_{model}(trace))}
                             n -> inf

        where:

            H(target) <= H(target, model)

        The finite-length approximation to the cross-entropy is then given by
        (eq. 3.51 from SLP ch3):

            H(trace) = (- 1 / N) log(P_{model}(trace))

        References:
        NLTK.lm.api
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      trace:  The sequence of symbols to evaluate
        :param      base:   The log base. Commonly set to 2 in classic
                            information theory literature

        :returns:   the approximate cross-entropy of the given trace
        """

        N = len(trace)

        return (-1.0 / N) * self.logscore(trace, base)

    def perplexity_approx(self, trace: Symbols, base: float = 2.0) -> float:
        """
        computes approximate perplexity of the given trace in the language of
        the PDFA

        The approximate perplexity is based on computing the approximate
        cross-entropy (cross_entropy_approximate) (eq. 3.52 of SLP).

        References:
        NLTK.lm.api
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      trace:  The sequence of symbols to evaluate
        :param      base:   The log base used for log probability calculations
                            of the approximate cross-entropy underpinning the
                            perplexity. Commonly set to 2 in classic
                            information theory literature

        :returns:   the approximate perplexity of the given trace
        """

        return base ** self.cross_entropy_approx(trace, base)

    def cross_entropy(
        self,
        traces: List[Symbols],
        actual_trace_probs: Probabilities,
        base: float = 2.0,
    ) -> float:
        """
        computes actual cross-entropy of the given traces in the language of
        the PDFA on the given actual trace probabilities

        References:
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      traces:              The list of sequences of symbols to
                                         evaluate the model's actual cross
                                         entropy on.
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :param      base:                The log base. Commonly set to 2 in
                                         classic information theory literature

        :returns:   the actual cross-entropy of the given trace
        """

        cross_entropy_sum = 0.0

        for target_p, trace in zip(actual_trace_probs, traces):
            est_q = self.score(trace)
            cross_entropy_sum += ylogx(y=target_p, x=est_q)

        return -cross_entropy_sum

    def perplexity(
        self,
        traces: List[Symbols],
        actual_trace_probs: Probabilities,
        base: float = 2.0,
    ) -> float:
        """
        computes actual perplexity of the given traces in the language of
        the PDFA on the given actual trace probabilities

        References:
        Speech and Language Processing (SLP), 3 ed., Ch3
        (https://web.stanford.edu/~jurafsky/slp3/3.pdf)

        :param      traces:              The list of sequences of symbols to
                                         evaluate the model's actual cross
                                         entropy on.
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :param      base:                The log base. Commonly set to 2 in
                                         classic information theory literature

        :returns:   the actual cross-entropy of the given trace
        """

        return base ** self.cross_entropy(traces, actual_trace_probs, base)

    def norm(
        self, traces: List[Symbols], actual_trace_probs: Probabilities, n: int = 2
    ) -> float:
        """
        computes Ln Norm (default n=2)

        :param      traces:              The list of sequences of symbols to
                                         evaluate the model's actual cross
                                         entropy on.
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :param      n:                   n norm

        :returns:   the ln norm between true and estimated distributions
        """

        norm_sum = 0.0

        for target_prob, trace in zip(actual_trace_probs, traces):
            p = self.score(trace)
            norm = abs(target_prob - p) ** n
            norm_sum += norm

        return norm_sum ** (1 / n)

    def average_norm(
        self, traces: List[Symbols], actual_trace_probs: Probabilities, n: int = 2
    ) -> float:
        """
        computes Ln Norm (default n=2)

        :param      traces:              The list of sequences of symbols to
                                         evaluate the model's actual cross
                                         entropy on.
        :param      actual_trace_probs:  The actual probability of each trace
                                         in the target language distribution
        :param      n:                   n norm

        :returns:   the average ln norm between true and estimated distributions
        """

        return 1 / len(traces) * self.norm(traces, actual_trace_probs, n)

    def kldivergence(
        self,
        traces: List[Symbols],
        actual_trace_probs: Probabilities,
        base=2.0,
        epsilon=0.001,
    ):
        """
        Forward KL Divergence
        Use ForwardKL on traces generated by this automaton.

        KL(p||q) =  p log(p) - p log(q)

        where p is the true probability and q is the estimated probability
        q must not be 0 otherwise the KL divergence goes to infinity.
        Therefore, we must be certain that the traces were generated by
        this automaton, so that q is always positive q>0
        """
        kldivergence = 0.0
        max_trace_length = max(list(map(len, traces)))

        for target_p, trace in zip(actual_trace_probs, traces):
            est_q = self.score(trace)
            if est_q == 0.0:
                est_q = epsilon**max_trace_length
            kldivergence += xlogx(target_p) - xlogy(x=target_p, y=est_q)
        return kldivergence

    def reverse_kldivergence(
        self,
        traces: List[Symbols],
        actual_trace_probs: Probabilities,
        base=2.0,
        epsilon=0.001,
    ):
        """
        Reverse KL Divergence
        Use ReverseKL on traces generated by other automaton

        KL(q||p) =  q log(q) - q log(p)

        where p is the true probability and q is the estimated probability
        p must not be 0 otherwise the KL divergence goes to infinity.
        Therefore, we must be certain that the traces were generated by
        the true automaton, so that p is always positive p>0
        """
        kldivergence = 0.0
        max_trace_length = max(list(map(len, traces)))

        for target_p, trace in zip(actual_trace_probs, traces):
            est_q = self.score(trace)
            if target_p == 0.0:
                target_p = epsilon**max_trace_length
            kldivergence += xlogx(est_q) - xlogy(x=est_q, y=target_p)
        return kldivergence

    def predictive_accuracy(
        self, test_traces: List[Symbols], pred_method: str = "max_prob"
    ) -> float:
        """
        compares the model's predictions to the actual values of the next
        symbol and returns the ratio of correct predictions.

        :param      test_traces:  The traces to compute predictive accuracy for
        :param      pred_method:  The method used to choose the next state.
                                  see _choose_next_state for details on how
                                  each pred_method is implemented.
                                  {'sample', 'max_prob'}

        :returns:   predictive accuracy ratio ([0 -> 1]) of the model on the
                    given traces
        """

        N = len(test_traces)
        num_correct_predictions = 0

        for trace in test_traces:
            observations = trace[:-1]
            actual_symbol = trace[-1]

            # check the predictive capability when conditioned on all but the
            # last symbol
            predicted_symbol = self.predict(observations, pred_method)

            if predicted_symbol == actual_symbol:
                num_correct_predictions += 1

        return num_correct_predictions / N

    def mdi_score(self, traces: List[Symbols]) -> float:
        """
        computes the mdi score given a list of traces and the current
        automata

        :param traces:                   The list of sequences of symbols to
                                         evaluate the model's MDI score.

        :returns:   the MDI score
        """
        # computes the occurrence of each trace
        n = len(traces)

        # computes the MDI score
        score = 0.0
        for trace in traces:
            score += (traces.count(trace) / n) * -self.logscore(trace)

        # divide by the No. of nodes in the Automata
        score = score / len(self.nodes)

        return score

    def refit_prob_dist(self, traces: List[Symbols]) -> None:
        transitionCounts = {}
        start_state = self.start_state

        for trace in traces:
            curr_state = start_state
            trace_prob = 1.0

            for symbol in trace:
                next_state, _ = self._get_next_state(curr_state, symbol)

                if curr_state not in transitionCounts:
                    transitionCounts[curr_state] = {}
                if symbol not in transitionCounts[curr_state]:
                    transitionCounts[curr_state][symbol] = 0

                transitionCounts[curr_state][symbol] += 1
                curr_state = next_state

        # Now compare against the actual distribution
        for curr_state, eachNodeTransitions in transitionCounts.items():
            sum_counts = sum(list(eachNodeTransitions.values()))
            for symbol, counts in eachNodeTransitions.items():
                next_state, old_prob_dist = self._get_next_state(curr_state, symbol)
                new_prob_dist = counts / sum_counts
                for _, edge_data in self[curr_state][next_state].items():
                    if edge_data["symbol"] == symbol:
                        edge_data["probability"] = new_prob_dist

        self._initialize_node_edge_properties(
            initial_weight_key=None,
            final_weight_key="final_probability",
            state_observation_key=None,
            can_have_accepting_nodes=True,
            edge_weight_key="probability",
            merge_sinks=False,
        )

    def _set_state_acceptance(self, curr_state: Node) -> None:
        """
        Sets the state acceptance property for the given state.

        If curr_state's final_probability >= self._beta, then the state accepts

        :param      curr_state:  The current state's node label
        """

        curr_final_prob = self._get_node_data(curr_state, "final_probability")

        if curr_final_prob >= self._beta:
            state_accepts = True
        else:
            state_accepts = False

        self._set_node_data(curr_state, "is_accepting", state_accepts)

    @classmethod
    def _compute_product(
        cls, specification: PDFA, safety: SafetyDFA, delete_sinks: bool
    ) -> dict:
        """
        Compute a product between a PDFA and a safetyDFA

        :param      specification:      A PDFA specification
        :param      safety:             The safety automaton instance
        :param      delete_sinks:       whether to delete unnecessary
                                        sink states

        :returns:   dict of configurations to build the pdfa
        """
        # naming to follow written algorithm
        C = specification
        S = safety

        symbols = set()
        # Visit all nodes and edges to find all symbols
        for node in S.nodes:
            for dest_state, edges in S[node].items():
                for edge_key, edge_data in edges.items():
                    partial_symbols = SafetyDFA._extract_symbols_from_formula(
                        edge_data["symbol"]
                    )
                    symbols |= partial_symbols

        # Initial State
        qc_init = C.start_state
        qs_init = S.start_state
        init_prod_state = cls._get_product_state_label(qc_init, qs_init)

        # Prepare dicts to store graph info
        nodes = {}
        edges = {}

        # Start Search from the initial state
        search_queue = queue.Queue()
        search_queue.put((qc_init, qs_init))
        visited = set()
        visited.add((qc_init, qs_init))

        nodes_to_delete = []
        inverted_edges = {}

        while not search_queue.empty():
            # pop a node
            qc, qs = search_queue.get()
            qc_final_prob = C._get_node_data(qc, "final_probability")
            prob_sum = qc_final_prob

            # Search each edge in Cosafety PDFA
            cosafe_sigmas = []
            for cosafe_edges in C[qc].values():
                for cosafe_edge in cosafe_edges.values():
                    cosafe_sigmas.append(cosafe_edge["symbol"])

            formulas = []
            for safe_edges in S[qs].values():
                for safe_edge in safe_edges.values():
                    formulas.append(safe_edge["symbol"])

            # Check if each edge in Cosafety PDFA satisfies
            # Safety Specification
            for cosafe_sigma in cosafe_sigmas:
                for formula in formulas:
                    # Check if the "sigma" satisfies the "formula"
                    valid = S.satisfy_formula(formula, cosafe_sigma, symbols)

                    # No need for searching for invalid transition
                    if not valid:
                        continue

                    qs_prime, _ = S._get_next_state(qs, formula)
                    # If the next state is a sink state, don't transition
                    if len(S[qs_prime].values()) == 0:
                        continue

                    # Get Transition Probability from Cosafety PDFA
                    qc_prime, trans_prob = C._get_next_state(qc, cosafe_sigma)
                    qc_prime_final_prob = C._get_node_data(
                        qc_prime, "final_probability"
                    )

                    (nodes, edges, _, _) = cls._add_product_edge(
                        nodes,
                        edges,
                        qc_src=qc,
                        qc_dest=qc_prime,
                        qs_src=qs,
                        qs_dest=qs_prime,
                        qc_src_final_prob=qc_final_prob,
                        qc_dest_final_prob=qc_prime_final_prob,
                        sigma=cosafe_sigma,
                        trans_prob=trans_prob,
                    )

                    inverted_edges = cls._add_product_inverted_edge(
                        inverted_edges,
                        qc_src=qc,
                        qc_dest=qc_prime,
                        qs_src=qs,
                        qs_dest=qs_prime,
                        qc_src_final_prob=qc_prime_final_prob,
                        qc_dest_final_prob=qc_final_prob,
                        sigma=cosafe_sigma,
                        trans_prob=trans_prob,
                    )

                    prob_sum += trans_prob

                    prod_dest_state = (qc_prime, qs_prime)
                    if prod_dest_state not in visited and qc_prime_final_prob != 1.0:
                        visited.add(prod_dest_state)
                        search_queue.put(prod_dest_state)

            if prob_sum == 0.0:
                src_state = cls._get_product_state_label(qc, qs)
                nodes_to_delete.append(src_state)

        if delete_sinks:
            nodes, edges = cls._minimize_sink_states(
                nodes, edges, inverted_edges, nodes_to_delete
            )

        return cls._package_data(specification, nodes, edges, init_prod_state)

    @classmethod
    def _minimize_sink_states(
        cls, nodes: dict, edges: dict, inverted_edges: dict, nodes_to_delete: list
    ):
        """
        Delete the unnecessary sink states

        :param      nodes:              dict of nodes to build the product
        :param      edges:              dict of edges to build the product
        :param      inverted_edges:     dict of edges between dest_node and
                                        src_node
        :param      nodes_to_delete:    List of sink nodes to be deleted

        :returns:   dict of nodes to build the pdfa,
                    dict of edges to build the pdfa
        """
        while len(nodes_to_delete) != 0:
            node = nodes_to_delete.pop()

            # delete outgoing edges
            outgoing_edges = edges[node]
            del edges[node]
            for dest_node, data in outgoing_edges.items():
                del inverted_edges[dest_node][node]

            # for each incoming edges
            incoming_edges = inverted_edges[node]

            for src_node, data in incoming_edges.items():
                # delete the edge
                del edges[src_node][node]

                # if not visited
                if src_node not in nodes_to_delete:
                    # get edges' probabilities
                    edge_probs = [
                        sum(data["probabilities"]) for data in edges[src_node].values()
                    ]

                    # if all src node's outgoing edges sum up to 0
                    # then append the node to delete list
                    if sum(edge_probs) == 0.0:
                        nodes_to_delete.append(src_node)

            # delete the node
            del nodes[node]

        return nodes, edges

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
    def _add_product_node(
        cls, nodes: dict, qc: Node, qs: Node, qc_final_prob: Probability
    ) -> Tuple[dict, Node]:
        """
        Adds a newly identified product state to the nodes dict w/ needed data

        :param      nodes:             dict of nodes to build the product out
                                       of. must be in the format needed by
                                       networkx.add_nodes_from()
        :param      qc:                state label in the dynamical system
        :param      qs:                state label in the specification
        :param      qc_final_prob:     the probability of terminating at q in
                                       the specification

        :returns:   nodes dict populated with all of the given data, and the
                    label of the newly added product state
        """

        prod_state = cls._get_product_state_label(qc, qs)
        is_violating = qc == SPEC_VIOLATING_STATE

        if prod_state not in nodes:
            prod_state_data = {
                "final_probability": qc_final_prob,
                "trans_distribution": None,
                "is_violating": is_violating,
                "is_accepting": None,
            }
            nodes[prod_state] = prod_state_data

        return nodes, prod_state

    @classmethod
    def _add_product_edge(
        cls,
        nodes: dict,
        edges: dict,
        qc_src: Node,
        qc_dest: Node,
        qs_src: Node,
        qs_dest: Node,
        qc_src_final_prob: Probability,
        qc_dest_final_prob: Probability,
        sigma: Symbol,
        trans_prob: Probability,
    ) -> Tuple[dict, dict, Node, Node]:
        """
        Adds a newly identified product edge to the nodes & edges dicts

        :param      nodes:              dict of nodes to build the product out
                                        of. Must be in the format needed by
                                        networkx.add_nodes_from()
        :param      edges:              dict of edges to build the product out
                                        of. Must be in the format needed by
                                        networkx.add_edges_from()
        :param      qc_src:             source product edge's dynamical system
                                        state
        :param      qc_dest:            dest. product edge's dynamical system
                                        state
        :param      qs_src:             source product edge's specification
                                        state
        :param      qs_dest:            dest. product edge's specification
                                        state
        :param      qc_src_final_prob:  the probability of terminating at q_src
                                        in the specification
        :param      qc_dest_final_prob: the probability of terminating at
                                        q_dest in the specification
        :param      sigma:              dynamical system control input symbol
                                        enabling the product edge
        :param      trans_prob:         The product edge's transition prob.

        :returns:   nodes dict populated w/ all the given data for src & dest
                    edges dict populated w/ all the given data,
                    the label of the newly added source product state,
                    the label of the newly added product state
        """

        nodes, prod_src = cls._add_product_node(
            nodes, qc_src, qs_src, qc_src_final_prob
        )
        nodes, prod_dest = cls._add_product_node(
            nodes, qc_dest, qs_dest, qc_dest_final_prob
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
    def _add_product_inverted_edge(
        cls,
        inverted_edges: dict,
        qc_src: Node,
        qc_dest: Node,
        qs_src: Node,
        qs_dest: Node,
        qc_src_final_prob: Probability,
        qc_dest_final_prob: Probability,
        sigma: Symbol,
        trans_prob: Probability,
    ):
        """
        Adds a newly identified product edge (reverse) to the inverted_edges
        dicts

        :param      inverted_edges:     dict of edges to build the product out
                                        of. Must be in the format needed by
                                        networkx.add_edges_from()
        :param      qc_src:             source product edge's dynamical system
                                        state
        :param      qc_dest:            dest. product edge's dynamical system
                                        state
        :param      qs_src:             source product edge's specification
                                        state
        :param      qs_dest:            dest. product edge's specification
                                        state
        :param      qc_src_final_prob:  the probability of terminating at q_src
                                        in the specification
        :param      qc_dest_final_prob: the probability of terminating at
                                        q_dest in the specification
        :param      sigma:              dynamical system control input symbol
                                        enabling the product edge
        :param      trans_prob:         The product edge's transition prob.

        :returns:   nodes dict populated w/ all the given data for src & dest
                    edges dict populated w/ all the given data,
                    the label of the newly added source product state,
                    the label of the newly added product state
        """
        prod_src = cls._get_product_state_label(qc_src, qs_src)
        prod_dest = cls._get_product_state_label(qc_dest, qs_dest)
        prod_edge_data = {"symbols": [sigma], "probabilities": [trans_prob]}
        prod_edge = {prod_src: prod_edge_data}

        if prod_dest in inverted_edges:
            if prod_src in inverted_edges[prod_dest]:
                existing_edge_data = inverted_edges[prod_dest][prod_src]
                existing_edge_data["symbols"].extend(prod_edge_data["symbols"])
                new_probs = prod_edge_data["probabilities"]
                existing_edge_data["probabilities"].extend(new_probs)

                inverted_edges[prod_dest][prod_src] = existing_edge_data
            else:
                inverted_edges[prod_dest].update(prod_edge)
        else:
            inverted_edges[prod_dest] = prod_edge

        return inverted_edges

    @classmethod
    def _package_data(
        cls, specification: PDFA, nodes: dict, edges: dict, init_prod_state: Node
    ) -> dict:
        """
        Delete the unnecessary sink states

        :param      nodes:              dict of nodes to build the product
        :param      edges:              dict of edges to build the product
        :param      init_prod_state:    initial node of the product

        :returns:   dict of nodes to build the pdfa,
                    dict of edges to build the pdfa
        """
        config_data = {}

        final_sym = specification.final_transition_sym
        config_data["final_transition_sym"] = final_sym
        empty_sym = specification.empty_transition_sym
        config_data["empty_transition_sym"] = empty_sym

        # can directly compute these from the graph data
        symbols = set()
        state_labels = set()
        for state, edge in edges.items():
            for _, edge_data in edge.items():
                symbols.update(edge_data["symbols"])
                state_labels.add(state)

        alphabet_size = len(symbols)
        num_states = len(state_labels)

        config_data["alphabet_size"] = alphabet_size
        config_data["num_states"] = num_states

        (symbol_display_map, nodes, edges) = Automaton._convert_states_edges(
            nodes, edges, final_sym, empty_sym, is_stochastic=IS_STOCHASTIC
        )
        config_data["nodes"] = nodes
        config_data["edges"] = edges
        config_data["start_state"] = init_prod_state
        config_data["symbol_display_map"] = symbol_display_map

        return config_data


class PDFABuilder(Builder):
    """
    Implements the generic automaton builder class for PDFA objects
    """

    def __init__(self) -> "PDFABuilder":
        """
        Constructs a new instance of the PDFABuilder
        """

        # need to call the super class constructor to gain its properties
        Builder.__init__(self)

        # keep these properties so we don't re-initialize unless underlying
        # data changes
        self.nodes = None
        self.edges = None

    def __call__(
        self, graph_data: {str, FDFA}, graph_data_format: str = "yaml", **kwargs: dict
    ) -> PDFA:
        """
        Returns an initialized PDFA instance given the graph_data

        graph_data and graph_data_format must match

        :param      graph_data:         The variable specifying graph data
        :param      graph_data_format:  The graph data file format.
                                        {'yaml', 'fdfa_object'}
        :param      kwargs:             The keywords arguments to the specific
                                        constructors

        :returns:   instance of an initialized PDFA object

        :raises     ValueError:         checks if graph_data and
                                        graph_data_format have a compatible
                                        data loader
        """

        if graph_data_format == "yaml":
            self._instance = self._from_yaml(graph_data, **kwargs)
        elif graph_data_format == "fdfa_object":
            self._instance = self._from_fdfa(graph_data, **kwargs)
        elif graph_data_format == "existing_objects":
            spec = graph_data[0]
            safety = graph_data[1]
            self._instance = self._from_automata(spec, safety, **kwargs)
        else:
            msg = (
                'graph_data_format ({}) must be one of: "yaml", '
                + '"fdfa_object"'.format()
            )
            raise ValueError(msg)

        return self._instance

    def _from_yaml(self, graph_data_file: str, is_normalized: bool = True) -> PDFA:
        """
        Returns an instance of a PDFA from the .yaml graph_data_file

        Only reads the config data once, otherwise just returns the built
        object

        :param      graph_data_file:  The graph configuration file name

        :returns:   instance of an initialized PDFA object

        :raises     ValueError:       checks if graph_data_file's ext is YAML
        """

        _, file_extension = os.path.splitext(graph_data_file)

        allowed_exts = [".yaml", ".yml"]
        if file_extension in allowed_exts:
            config_data = self.load_YAML_config_data(graph_data_file)
        else:
            msg = "graph_data_file ({}) is not a ({}) file"
            raise ValueError(msg.format(graph_data_file, allowed_exts))

        nodes_have_changed = self.nodes != config_data["nodes"]
        edges_have_changed = self.edges != config_data["edges"]
        no_instance_loaded_yet = self._instance is None

        if no_instance_loaded_yet or nodes_have_changed or edges_have_changed:
            # nodes and edges must be in the format needed by:
            #   - networkx.add_nodes_from()
            #   - networkx.add_edges_from()
            final_transition_sym = config_data["final_transition_sym"]
            empty_transition_sym = config_data["empty_transition_sym"]
            (symbol_display_map, states, edges) = Automaton._convert_states_edges(
                config_data["nodes"],
                config_data["edges"],
                final_transition_sym,
                empty_transition_sym,
                is_stochastic=True,
            )
            config_data["symbol_display_map"] = symbol_display_map
            config_data["nodes"] = states
            config_data["edges"] = edges
            config_data["is_normalized"] = is_normalized

            # saving these so we can just return initialized instances if the
            # underlying data has not changed
            self.nodes = states
            self.edges = edges

            instance = PDFA(**config_data)

            return instance

        return self._instance

    def _from_fdfa(
        self,
        fdfa: FDFA,
        merge_sinks: bool = False,
        smooth_transitions: bool = False,
        smoothing_amount: float = SMOOTHING_AMOUNT,
    ) -> PDFA:
        """
        Returns an instance of a PDFA from an instance of FDFA

        :param      fdfa:                initialized fdfa instance to convert
                                         to a pdfa
        :param      merge_sinks:         whether to combine all states
                                         together that have no outgoing
                                         edges
        :param      smooth_transitions:  whether or not to smooth the input
                                         sym. transition distributions
        :param      smoothing_amount:    probability mass to re-assign to
                                         unseen symbols at each node

        :returns:   instance of an initialized PDFA object
        """

        nodes, edges = fdfa.to_pdfa_data()

        # saving these so we can just return initialized instances if the
        # underlying data has not changed
        self.nodes = nodes
        self.edges = edges

        instance = PDFA(
            nodes=nodes,
            edges=edges,
            symbol_display_map=fdfa._symbol_display_map,
            # just choose a default value, FDFAs have no notion of acceptance
            # this at the moment
            beta=0.95,
            alphabet_size=fdfa.alphabet_size,
            num_states=fdfa.num_states,
            final_transition_sym=fdfa.final_transition_sym,
            empty_transition_sym=fdfa.empty_transition_sym,
            start_state=fdfa.start_state,
            smooth_transitions=smooth_transitions,
            smoothing_amount=smoothing_amount,
            merge_sinks=merge_sinks,
            is_normalized=True,
        )

        return instance

    def _from_automata(
        self,
        specification: PDFA,
        safety: SafetyDFA,
        smooth_transitions: bool = False,
        normalize_trans_probabilities: bool = False,
        delete_sinks: bool = True,
    ) -> PDFA:
        """
        Returns an instance of a PDFA from an instance of PDFA and DFA

        :param      specification:      The specification automaton instance
        :param      safety:             The safety automaton instance

        :returns:   instance of an initialized PDFA object
        """
        config_data = PDFA._compute_product(specification, safety, delete_sinks)

        config_data["smooth_transitions"] = smooth_transitions
        config_data["is_normalized"] = normalize_trans_probabilities

        self.nodes = config_data["nodes"]
        self.edges = config_data["edges"]

        if not self.edges:
            msg = "no compatible edges were found, so the product is empty"
            warnings.warn(msg, RuntimeWarning)
            instance = None
        else:
            instance = PDFA(**config_data)

        return instance
