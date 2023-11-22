from abc import ABCMeta, abstractmethod
from typing import List

from specless.typing import TimedTrace, Trace


class Specification(metaclass=ABCMeta):
    """Base class for all specification models

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def satisfy(self, demonstration) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError()


class TimedPartialOrder(Specification):
    """Timed Partial Order Models"""

    def __init__(self) -> None:
        super().__init__()

    def satisfy(self, demonstration: TimedTrace) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (TimedTrace): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError()


class AutomataSpecification(Specification):
    """Base class for all Automata-based Specification Models"""

    def __init__(self) -> None:
        super().__init__()


class DFA(AutomataSpecification):
    """Deterministic Finite Automata Models"""

    def __init__(self) -> None:
        super().__init__()

    def satisfy(self, demonstration: Trace) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (Trace): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError()


class PDFA(AutomataSpecification):
    """Probabilistic Deterministic Finite Automata Models"""

    def __init__(self) -> None:
        super().__init__()

    def satisfy(self, demonstration: Trace) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (Trace): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError()


class MultiSpecifications(Specification):
    """Class for maintaining multiple specification models
    to make it work as a single specification model
    """

    def __init__(self, specifications: List[Specification]) -> None:
        super().__init__()
        self.specifications: List[Specification] = specifications

    def satisfy(self, demonstration) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError()
