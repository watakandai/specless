from abc import ABCMeta, abstractmethod

from .specification import Specification


class InferenceAlgorithm(metaclass=ABCMeta):
    """Base class for inference algorithms.
    The algorithm infers a specification from demonstrations.
    """

    # class variable shared by all instances
    @abstractmethod
    def __init__(self) -> None:
        pass  # instance variable unique to each instance

    @abstractmethod
    def infer(self, X) -> Specification:
        s = Specification()
        return s
