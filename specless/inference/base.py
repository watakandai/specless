"""
Inference Algorithm
===================
Inference algorithms then use such demonstrations to come up with a specification.
>> import specless as sl
>> traces = [[a,b,c], [a,b,b,c], [a,a,b,b,c]]
>> dataset = sl.ArrayDataset(traces)
>> inference = sl.TPOInference()
>> specification = inference.infer(demonstrations)
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
