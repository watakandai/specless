import networkx as nx
import pydot
from ltlf2dfa.parser.ltlf import LTLfParser as OriginalLTLfParser

from specless.specification.dfa import DFA


# NOTE: DEPRECATED
class LTLfParser:
    """LTLf Parser
    It parses a LTLf formula and translate it into a DFA
    """

    def __init__(self, engine: str = "ltlf2dfa") -> None:
        self.engine: str = engine

    def parse(self, formula_str: str) -> DFA:
        """Parse a fomula and translate it into a DFA

        Args:
            formula (str): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            DFA: _description_
        """
        if self.engine == "ltlf2dfa":
            parser = OriginalLTLfParser()
            dfa = parser(formula_str)
            dot = dfa.to_dfa(mona_dfa_out=False)
            print(dot)
            # TODO: Export it as a dot file
            P_list = pydot.graph_from_dot_data(dot)
            # Convert only the first such instance into a NetworkX graph.
            g = nx.drawing.nx_pydot.from_pydot(P_list[0])

            for n in g.nodes(data=True):
                print(n)
            for e in g.edges(data=True):
                print(e)

            return DFA(g)
        elif self.engine == "spot":
            parser = lambda x: x
            return DFA()
        else:
            raise Exception(f"No such engine called {self.engine}")
