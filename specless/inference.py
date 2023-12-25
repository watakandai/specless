"""
Inference Algorithm
===================
Inference algorithms then use such demonstrations to come up with a specification.
>> import specless as sl
>> traces = [[a,b,c], [a,b,b,c], [a,a,b,b,c]]
>> dataset = sl.TraceDataset(traces)
>> inference = sl.TPOInference()
>> specification = inference.infer(demonstrations)
"""
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Type

from specless.dataset import Dataset, TimedTraceDataset, TraceDataset
from specless.specification import DFA, Specification, TimedPartialOrder


class InferenceAlgorithm(metaclass=ABCMeta):
    """Base class for the inference algorithms.
    The algorithm infers a specification from demonstrations (dataset).
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args: tuple = args
        self.kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def infer(self, dataset: Type[Dataset]) -> Specification:
        raise NotImplementedError()


class TPOInferenceAlgorithm(InferenceAlgorithm):
    """The inference algorithm for inferring a TPO from a list of TimedTraces.

    Args:
        InferenceAlgorithm (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    def infer(self, dataset: Type[Dataset]) -> Specification:
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
            dataset (Type[Dataset]):        Timed Trace Data

        Raises:
            NotImplementedError: _description_

        Returns:
            Specification:                  Timed Partial Order
        """

        error_msg = "This class only takes dataset of type TimedTraceDataset"
        assert isinstance(dataset, TimedTraceDataset), error_msg

        dataset.apply(lambda data: data.sort_by("timestamp", inplace=False))
        traces: List = dataset.to_list(key="symbol")

        # Find a partial order
        partial_order_dict: Dict = self.find_partial_order_from_traces(traces)
        # Infer Timing Constraints
        global_constraints, local_constraints = self.infer_time_constraints(
            traces, partial_order_dict
        )

        return TimedPartialOrder(global_constraints, local_constraints)


class AutomataInferenceAlgorithm(InferenceAlgorithm):
    """The inference algorithm for inferring an automaton from a list of Traces,
    where trace is defined as a sequence of symbols, i.e. a set of strings.
    For example, ${a, b, c}$

    Args:
        InferenceAlgorithm (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    def infer(self, dataset: Type[Dataset]) -> Specification:
        error_msg = "This class only takes dataset of type TraceDataset"
        assert isinstance(dataset, TraceDataset), error_msg

        raise NotImplementedError()

        return DFA()
