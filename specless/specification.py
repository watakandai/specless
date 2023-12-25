from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx

from specless.typing import TimedTrace, Trace


# TODO: Inherit from two classes.
class Specification(nx.MultiDiGraph, metaclass=ABCMeta):
    """Base class for all specification models

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    def __init__(
        self,
        global_constraints: Dict[int, Tuple[float, float]] = {},
        local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = {},
    ) -> None:
        super().__init__()
        """
        Args:
            local_constraints: Forward Timing Constraints. (i, i+1) => (lb, ub)

        For example:
            local_constraints = {
                (1, 2): (5, 10)
                (1, 3): (5, 15)
                (3, 4): (0, 5)
                ...
            }
            tpo = TPO(local_constraints)
        """
        # Convert global constraints to type Dict[int, Dict[str, float]]
        global_constraints_: Dict = {}
        for node, bound_ in global_constraints.items():
            if not isinstance(bound_[0], (int, float)) or bound_[0] < 0:
                bound: tuple[float, float] = (0, bound_[1])
            if not isinstance(bound_[1], (int, float)) or bound_[1] < 0:
                bound = (bound_[0], float("inf"))
            if bound_[1] < bound_[0]:
                raise Exception("Upper bound must be greater than the lower bound")
            global_constraints_[node] = {"lb": bound[0], "ub": bound[1]}
        self.global_constraints: dict = global_constraints_

        # Convert local constraints to type Dict[int, Dict[int, Dict[str, float]]]
        # ex.) {Node U: {Node V: {LB: 0, UB: 10}}}
        local_constraints_: defaultdict = defaultdict(lambda: {})
        for k, bound in local_constraints.items():
            if not isinstance(bound[0], (int, float)) or bound[0] < 0:
                bound = (0, bound[1])
            if not isinstance(bound[1], (int, float)) or bound[1] < 0:
                bound = (bound[0], float("inf"))
            if bound[1] < bound[0]:
                raise Exception("Upper bound must be greater than the lower bound")
            local_constraints_[k[0]][k[1]] = {"lb": bound[0], "ub": bound[1]}
        self.local_constraints = dict(local_constraints_)

        # Store Reverse Constraints for the "satisfy" function to easily access the constraint
        # ex.) {Node V: {Node U: {LB: 0, UB: 10}}
        reverse_constraints_: defaultdict = defaultdict(lambda: {})
        for src, d in self.local_constraints.items():
            for tgt, bound in d.items():
                reverse_constraints_[tgt][src] = bound
        self.reverse_constraints: defaultdict = reverse_constraints_

        # For the sake of creating edges
        super().__init__(dict(local_constraints_))
        # Reduce the redundant edges
        g = nx.transitive_reduction(self)
        self.__dict__.update(g.__dict__)

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
