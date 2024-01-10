# 3rd-party packages
import copy
import heapq
import queue
import warnings
from typing import Callable, Iterable, List, Set, Tuple

import bidict
import numpy as np
from tqdm.auto import tqdm

# local packages
from .types import Node, Probabilities, Probability, Symbol, Symbols
from .utils import MaxHeap, MinHeap

Heap = List
InverseProbability = Probability
ViableStringsHeap = Heap[Tuple[InverseProbability, Tuple[Symbols, Probabilities]]]
MPSReturnData = Tuple[Symbols, Probability, ViableStringsHeap]


def should_use_BMPS_exact(
    num_strings_to_find: int, try_to_use_greedy: bool, is_deterministic: bool
) -> bool:
    """
    Determine if we should use BMPS_exact based on the type of automaton we
    have and what we actually want to solve for

    :param      num_strings_to_find:  The number of viable strings to return.
                                      only BMPS_exact can acutually sample
                                      strings, so setting this > 1 guarantees
                                      the use of BMPS_exact.
    :param      try_to_use_greedy:    whether to try using the MUCH faster
                                      greedy search algorithm. only possible if
                                      the automaton has deterministic
                                      transitions.
    :param      is_deterministic:     if the automaton is deterministic.
                                      SWDFA_MPS only works on deterministic
                                      automata.

    :returns:   whether to use the BMPS_exact solver or the SWDFA_MPS solver
    """

    if try_to_use_greedy:
        if is_deterministic:
            use_BMPS_exact = False
        else:
            use_BMPS_exact = True
    else:
        use_BMPS_exact = True

    return use_BMPS_exact


def postprocess_MPS(
    mps_symbols: Iterable[int],
    mps_prob: Probability,
    viable_strings: ViableStringsHeap,
    idx_to_symbol: Callable,
    use_BMPS_exact: bool,
    allow_empty_symbol: bool,
    backwards_search: bool,
) -> MPSReturnData:
    """
    Post-processes the results from both MPS routines.

    :param      mps_symbols:         The symbols for the MPS
    :param      mps_prob:            The probability of the MPS
    :param      viable_strings:      Max heap containing all "viable" MPS
                                     candididates
    :param      idx_to_symbol:       The mapping from symbol index to symbol
    :param      use_BMPS_exact:      whether to use BMPS_exact
    :param      allow_empty_symbol:  Indicates if the empty symbol is allowed
                                     to be considered the MPS
    :param      backwards_search:    Whether to search from the with final
                                     probability back to the start state. Often
                                     will improve performance.

    :returns:   most probable string, probability of producing the most
                probable string, num_strings_to_find (their probs., viable
                strings) ranked by each string's probability
    """

    def process_string(symbols):
        if symbols:
            # first symbol will always be the empty symbol, so remove it if we
            # don't want it (typically we don't)
            if not allow_empty_symbol:
                symbols = symbols[1:]

            # BMPS_exact treats symbols only as integers indices, so
            # we start by converting the symbol indices back to actual symbols
            if use_BMPS_exact:
                # if we ran the search from all final states to the start
                # state, we must now reverse the order of each sequence of
                # symbols
                if backwards_search:
                    symbols.reverse()

                symbols = idx_to_symbol(symbols)

        else:
            symbols = None

        return symbols

    mps_symbols = process_string(mps_symbols)

    # need to apply the same sort of post-processing to each string in the heap
    if mps_symbols:
        new_viable_strings = MaxHeap()

        for prob, string in viable_strings:
            string = process_string(string)
            new_viable_strings.heappush((prob, string))

        viable_strings = new_viable_strings

    else:
        viable_strings = None

    return mps_symbols, mps_prob, viable_strings


def BMPS_search_step(
    string: List[int],
    symbol: Symbol,
    state_probabilities: np.ndarray,
    max_string_length: int,
    min_string_prob: Probability,
    num_strings_to_find: int,
    search_heap: Heap,
    viable_strings: Heap,
    seen: set,
    M: np.ndarray,
    F: np.ndarray,
    one_vec: np.ndarray,
    viable_str_probabilities: set,
    add_entropy: bool,
    pbar: tqdm,
):
    """
    Makes one step of BMPS_exact search, updating all data structures.

    :param      string:                    The previous search string
    :param      symbol:                    The current symbol to try
    :param      state_probabilities:       The probability of being in each
                                           state under the current string
    :param      max_string_length:         The maximum viable string length
    :param      min_string_prob:           The minimum viable string
                                           probability
    :param      num_strings_to_find:       The number of strings to find
    :param      search_heap:               The string search heap
    :param      viable_strings:            The collection of viable strings,
                                           meeting all requirements to be a
                                           bmps string.
    :param      seen:                      collection of strings already
                                           searched
    :param      M:                         3D Matrix with all transition
                                           matrices, keyed on each symbol in
                                           the 2nd axis
    :param      F:                         probability of terminating in each
                                           state
    :param      one_vec:                   d x 1 vector of ones
    :param      viable_str_probabilities:  the set of probabilities of each
                                           viable string found thus far. needed
                                           for adding entropy.
    :param      add_entropy:               indicates if we should add a
                                           normally viable string if it has the
                                           same sting that we have already
                                           identified as "viable"
    :param      pbar:                      tqdm progress bar

    :returns:   Updated data structures and if we have enough viable strings
    """

    # need to make a copy here so we don't add invalid symbols to the search
    string_new = string.copy()
    string_new.append(symbol)

    # apply the symbol to the automaton and see where you would end up, and
    # with what probability
    state_probabilities_new = state_probabilities @ M[:, :, symbol]
    new_string_prob = (state_probabilities_new @ F).item()
    is_viable_string = new_string_prob > min_string_prob

    if add_entropy:
        same_probability = new_string_prob in viable_str_probabilities
        is_viable_string = is_viable_string and not same_probability

    if is_viable_string:
        heap_item = string_new
        heap_weight = new_string_prob
        viable_strings.heappush((heap_weight, heap_item))

        if add_entropy:
            viable_str_probabilities.add(new_string_prob)

        # if the string is viable, we can actually return it. We will always
        # return the most probable string encountered thus far if we only want
        # to return one string, and we will otherwise add the new viable string
        # to the set of viable strings
        n_viable_str = len(viable_strings)
        have_enough_strings = n_viable_str == num_strings_to_find

        pbar.update(1)
    else:
        have_enough_strings = False
        pbar.update(0)

    # only keep a possible new symbol if it's (non-final) emission probability
    # is above the minimum probability threshold and the string isn't too long
    curr_emis_prob = (state_probabilities_new @ one_vec).item()
    string_still_viable = curr_emis_prob > min_string_prob
    hasnt_terminated = new_string_prob < np.finfo(float).eps

    string_could_be_mps = hasnt_terminated and string_still_viable
    string_length_below_bound = len(string) < max_string_length
    is_new = tuple(string_new) not in seen

    if string_length_below_bound and string_could_be_mps and is_new:
        seen.add(tuple(string_new))
        heap_item = (string_new, state_probabilities_new)
        heap_weight = curr_emis_prob
        search_heap.heappush((heap_weight, heap_item))

    return (
        search_heap,
        viable_strings,
        seen,
        viable_str_probabilities,
        have_enough_strings,
    )


def BMPS_exact(
    symbols: List[int],
    M: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    d: int,
    empty_symbol: int,
    min_string_prob: Probability,
    max_string_length: int,
    num_strings_to_find: int = 1,
    depth_first: bool = False,
    add_entropy: bool = True,
    disable_pbar: bool = False,
) -> MPSReturnData:
    """
    Finds the bounded, most probable string(s) (MPS) in a stochastically
    weighted finite automaton (SWFA).

    Automaton MUST have edge weights as transition probabilities, but the
    outgoing transition weights don't necessarily need to add up to 1, i.e. the
    transition matrices don't need to be formal Stochastic Matrices.

    This is useful if the automaton is a product, and thus its MPS can be
    projected onto its constituent automaton and have the same probability in
    its constituent automaton.

    Will default to return only the MOST probable, viable string thus far, but
    this algorithm generalizes to return the num_strings_to_find viable
    strings, decreasingly sorted by their probabilities.

    Originally written as BMPS_exact in:
    "The most probable string: an algorithmic study" by de la Higuera et. al

    :param      symbols:              All symbol indices in the symbol map
    :param      M:                    a (d x d x num_symbols) tensor containing
                                      the probabilistically weighted (NOT
                                      NECESSARILY stochastic) transition
                                      matrices representing the automaton,
                                      keyed on the third index - i.e. by symbol
    :param      S:                    a (1 x d) vector containing the initial
                                      state probabilities
    :param      F:                    a (d x 1) vector containing the final
                                      state probabilities
    :param      d:                    the number of states in the automaton
    :param      empty_symbol:         The "empty" symbol
    :param      min_string_prob:      The minimum string probability
    :param      max_string_length:    The maximum string length
    :param      num_strings_to_find:  The number of viable strings to return.
                                      Defaults to only return the ONE, highest
                                      probability string encountered thus far
                                      in the search, which means the algorithm
                                      is the original BMPS_exact. If >1, then
                                      the algorithm returns the
                                      num_strings_to_find most probable, viable
                                      strings from the search heap.
    :param      depth_first:          Whether to explore the automaton using a
                                      depth-first search pattern. Using a
                                      depth-first search pattern will be faster
                                      for very deep, tree-shaped automaton, but
                                      will not return the absolute best symbol
                                      sequence for the given min_string_prob
                                      and max_string_length. Only turn on if
                                      you have a terminal states deep in the
                                      automaton and you need the search to be
                                      faster.
    :param      add_entropy:          Only keeps a new viable string if it has
                                      a previously unseen probability of being
                                      generated
    :param      disable_pbar:         Disable pbar for speeding up the
                                      computation speed.

    :returns:   (most probable word in the SWFA, it's probability,
                num_strings_to_find viable strings in a max heap container)
    """

    # if we want depth-first search, then make the search heap a min heap so
    # it explores deeper in the automaton (lower probability traces) first.
    #
    # if we want breadth-first search, then make the search heap a max heap so
    # it explores more shallowly in the automaton (higher probability traces)
    # first.
    if depth_first:
        search_heap = MinHeap()
    else:
        search_heap = MaxHeap()

    # viable_strings are in a max heap keyed on viable string probability,
    # as we want to return the "best" strings ranked in descending string
    # probability
    viable_strings = MaxHeap()
    seen = set()
    viable_str_probs = set()

    string = [empty_symbol]
    p_empty = (S @ F).item()
    if p_empty > min_string_prob:
        return string, p_empty, viable_strings

    search_heap.heappush((p_empty, (string, S)))
    one_vec = np.ones(shape=(d, 1))

    with tqdm(total=num_strings_to_find, disable=disable_pbar) as pbar:
        while search_heap:
            _, (string, state_probabilities) = search_heap.heappop()

            for symbol in symbols:
                (
                    search_heap,
                    viable_strings,
                    seen,
                    viable_str_probs,
                    have_enough_strings,
                ) = BMPS_search_step(
                    string,
                    symbol,
                    state_probabilities,
                    max_string_length,
                    min_string_prob,
                    num_strings_to_find,
                    search_heap,
                    viable_strings,
                    seen,
                    M,
                    F,
                    one_vec,
                    viable_str_probs,
                    add_entropy,
                    pbar,
                )

                # the MPS is not always the most recently found string, so we
                # instead are just going to view the first element of the max
                # heap of viable strings, as this should be the highest
                # probability viable string.
                if have_enough_strings:
                    mps_probability, mps = copy.deepcopy(viable_strings[0])
                    return mps, mps_probability, viable_strings

    pbar.close()

    # no viable string found OR we only found < num_strings_to_find viable strs
    if viable_strings:
        mps_probability, mps = copy.deepcopy(viable_strings[0])

        n_viable_str = len(viable_strings)
        if n_viable_str < num_strings_to_find:
            msg = (
                f"only found {n_viable_str} viable strings when "
                + f"{num_strings_to_find} were requested. Returning "
                + "partially full heap of viable strings. Try lowering "
                + f"min_string_prob ({min_string_prob}) or increasing "
                + f"max_string_length ({max_string_length}) to find more."
            )

            warnings.warn(msg, RuntimeWarning)

        return mps, mps_probability, viable_strings
    else:
        return None, None, None


def SWDFA_MPS(
    states: Set[Node],
    start_state: Node,
    F: np.ndarray,
    empty_symbol: Symbol,
    node_index_map: bidict,
    trans_prob_fcn: Callable,
    transition_map: Callable,
) -> MPSReturnData:
    """
    Computes the EXACT consensus string (the actual most probable string (MPS))
    for a stochastically weighted DETERMINISTIC finite automaton (SWDFA).

    :warning THIS MPS CALCULATION IS ONLY VALID FOR A DETERMINISTIC AUTOMATA,
             as this is a very fast greedy algorithm where the optimal
             substructure assumption only holds for deterministic automata.

    :param      states:          The states of the automaton
    :param      start_state:     The start state of the SWDFA
    :param      F:               a (|states| x 1) vector containing the final
                                 state probabilities
    :param      empty_symbol:    The empty symbol
    :param      node_index_map:  a mapping from node label to it's index in the
                                 vectorized representation of the automaton
    :param      trans_prob_fcn:  a function that extracts the transition
                                 probabilities and associated symbols at the
                                 current state.
    :param      transition_map:  a map of start state label and symbol to
                                 destination state

    :returns:   (most probable word in the SWDFA, it's probability,
                 ALL viable strings in a max heap container)
    """

    search_queue = queue.Queue()

    init_state = start_state
    symbols, trans_probs = trans_prob_fcn(init_state)

    best_symbols = {state: [empty_symbol] for state in states}
    best_state_probs = {state: 0.0 for state in states}
    best_state_probs[init_state] = 1.0

    maxHeap = []

    # push the root node
    heapq.heappush(maxHeap, (1.0, init_state))

    with tqdm(total=len(states) - 1) as pbar:
        while len(maxHeap) != 0:
            probability, src_state = heapq._heappop_max(maxHeap)

            # check if repetitive visit has larger probability
            if probability < best_state_probs[src_state]:
                continue

            symbols, trans_probs = trans_prob_fcn(src_state)

            for symbol, trans_prob in zip(symbols, trans_probs):
                dest_state = transition_map[(src_state, symbol)]
                new_dest_prob = probability * trans_prob

                if new_dest_prob > best_state_probs[dest_state]:
                    best_state_probs[dest_state] = new_dest_prob
                    heapq.heappush(maxHeap, (new_dest_prob, dest_state))
                    # best_inverse_transitions[dest_state] = src_state
                    best_symbols[dest_state] = best_symbols[src_state].copy()
                    best_symbols[dest_state].append(symbol)
                    pbar.update(1)
                else:
                    pbar.update(0)

    pbar.close()

    # terminate all of the strings, and then find the best one using a max heap
    # sort, as we also want to return this heap
    viable_strings = MaxHeap()

    for idx, term_prob in enumerate(F.flatten()):
        state = node_index_map.inv[idx]
        best_state_probs[state] *= term_prob
        string_prob = best_state_probs[state]

        if string_prob > 0:
            string = best_symbols[state]
            viable_strings.heappush((string_prob, string))

    # need to check if we found any non-zero prob. strings
    if viable_strings:
        mps_probability, mps = copy.deepcopy(viable_strings[0])
        return mps, mps_probability, viable_strings
    else:
        # no viable string found
        return None, None, None
