"""
===================
Inference Algorithm
===================

Inference algorithms then use such demonstrations to come up with a specification.

Examples
--------

>>> import specless as sl
>>> demonstrations: list = [
...     [[1, "a"], [2, "b"], [3, "c"]],
...     [[4, "d"], [5, "e"], [6, "f"]],
... ]
>>> columns: list = ["timestamp", "symbol"]
>>> timedtrace_dataset = sl.ArrayDataset(demonstrations, columns)
>>> inference = sl.TPOInferenceAlgorithm()
>>> specification = inference.infer(timedtrace_dataset)

"""

from abc import ABCMeta, abstractmethod
from typing import Any, Union

from specless.dataset import BaseDataset
from specless.specification.base import Specification


class InferenceAlgorithm(metaclass=ABCMeta):
    """Base class for the inference algorithms.
    The algorithm infers a specification from demonstrations (dataset).
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args: tuple = args
        self.kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def infer(self, dataset: BaseDataset) -> Union[Specification, Exception]:
        raise NotImplementedError()
