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
>>> inference = sl.TPOInferenceAlgorithm()
>>> specification = inference.infer(demonstrations)

"""

from abc import ABCMeta, abstractmethod
from typing import Any, Union
from typing import Generic, TypeVar

from specless.dataset import BaseDataset
from specless.specification.base import Specification

# Define a generic type variable
T = TypeVar('T')

class InferenceAlgorithm(Generic[T], metaclass=ABCMeta):
    """Base class for the inference algorithms.
    The algorithm infers a specification from demonstrations (dataset).
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args: tuple = args
        self.kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def infer(self, T) -> Union[Specification, Exception]:
        raise NotImplementedError()
