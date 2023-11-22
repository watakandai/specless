from abc import ABCMeta, abstractmethod
from typing import Any, Type

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
        error_msg = "This class only takes dataset of type TimedTraceDataset"
        assert isinstance(dataset, TimedTraceDataset), error_msg

        raise NotImplementedError()

        return TimedPartialOrder()


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
