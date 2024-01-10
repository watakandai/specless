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
from collections import defaultdict
from typing import Any, Dict, List, Union

from specless.dataset import BaseDataset
from specless.inference.base import InferenceAlgorithm
from specless.specification.base import Specification
from specless.specification.partial_order import PartialOrder


class POInferenceAlgorithm(InferenceAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_partial_order(traces: List[List[str]]) -> Dict[str, List[str]]:
        forwards: defaultdict[Any, list[Any]] = defaultdict(lambda: [])
        backwards: defaultdict[Any, list[Any]] = defaultdict(lambda: [])

        # find forwards and negative backwards
        for trace in traces:
            visited: list = []
            for i, symbol in enumerate(trace):
                if i != 0:
                    for v in visited:
                        if symbol not in forwards[v]:
                            forwards[v].append(symbol)
                        if v not in backwards[symbol]:
                            backwards[symbol].append(v)
                visited.append(symbol)

        return {
            symbol: [s for s in forwards[symbol] if s not in backwards[symbol]]
            for symbol in forwards.keys()
        }

    def infer(self, dataset: BaseDataset) -> Union[Specification, Exception]:
        traces: List = dataset.tolist(key="symbol")
        partial_order = POInferenceAlgorithm.get_partial_order(traces)

        # add edges
        edges = []
        for symbol, next_symbols in partial_order.items():
            for next_symbol in next_symbols:
                edges.append((symbol, next_symbol))

        po = PartialOrder(edges)
        # It could
        # 1. TODO: include redundant edges
        # po = nx.transitive_reduction(po)
        # 2. be consisted of multiple independent graphs

        # Finally, return the partial order(s)
        return po
