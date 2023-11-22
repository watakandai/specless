from specless.specification import DFA


class LTLfParser:
    """LTLf Parser
    It parses a LTLf formula and translate it into a DFA
    """

    def __init__(self, engine: str) -> None:
        self.engine: str = engine

    def parse(self, formula: str) -> DFA:
        """Parse a fomula and translate it into a DFA

        Args:
            formula (str): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            DFA: _description_
        """
        raise NotImplementedError()
