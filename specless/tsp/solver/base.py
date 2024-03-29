from abc import ABCMeta, abstractmethod
from typing import List, Tuple


class TSPSolver(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, tsp) -> Tuple[List, float]:
        pass


class TSPWithTPOSolver(metaclass=ABCMeta):
    @abstractmethod
    def solve(self, tsp) -> Tuple[List, float]:
        pass
