from abc import ABCMeta, abstractmethod
from typing import List

import networkx as nx


class Specification(nx.DiGraph, metaclass=ABCMeta):
    """Base class for all specification models

    Args:
        metaclass (_type_, optional): _description_. Defaults to ABCMeta.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def satisfy(self, demonstration: List) -> bool:
        """Checks if a given demonstration satisfies the specification

        Args:
            demonstration (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            bool: _description_
        """
        raise NotImplementedError()


class AutomataSpecification(Specification):
    """Base class for all Automata-based Specification Models"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
