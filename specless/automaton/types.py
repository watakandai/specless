from typing import Hashable, List, Tuple, Iterable


# define these type defs for method annotation type hints
NXNodeList = List[Tuple[Hashable, dict]]
NXEdgeList = List[Tuple[Hashable, Hashable, dict]]

Node = Hashable
Observation = Hashable
Symbol = Hashable
Weight = int
Probability = float

Nodes = Iterable[Node]
Observations = Iterable[Observation]
Symbols = Iterable[Symbol]
Weights = Iterable[Weight]
Probabilities = Iterable[Probability]

Trans_data = (Weights, Nodes, Symbols)
SampledTransData = (Node, Symbol, Probability)
GeneratedTraceData = (List[Symbols], List[int], Probabilities)
