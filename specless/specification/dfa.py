from specless.specification.base import AutomataSpecification
from specless.typing import Trace


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
