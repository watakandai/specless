from abc import ABCMeta
from typing import Any

import gymnasium as gym

from specless.specification import Specification
from specless.strategy import MemorylessStrategy, PlanStrategy, PolicyStrategy, Strategy


class SynthesisAlgorithm(metaclass=ABCMeta):
    """Base classs for all synthesis algorithms"""

    def __init__(self) -> None:
        pass

    def synthesize(self, specification: Specification, env: gym.Env) -> Strategy:
        """Synthesizes a strategy in an env given the specification.

        Args:
            specification (Specification): _description_
            env (gym.Env): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Strategy: _description_
        """
        raise NotImplementedError()


class ProductGraphSynthesisAlgorithm(SynthesisAlgorithm):
    """Product Graph based synthesis algorithm"""

    def __init__(self) -> None:
        super().__init__()
        pass

    def synthesize(
        self, specification: Specification, env: gym.Env
    ) -> MemorylessStrategy:
        """Synthesizes a MemorylessStrategy in an env given the specification.

        Args:
            specification (Specification): _description_
            env (gym.Env): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Strategy: _description_
        """
        raise NotImplementedError()


class TSPSynthesisAlgorithm(SynthesisAlgorithm):
    """Traveling Salesman Problem based synthesis algorithm"""

    def __init__(self) -> None:
        super().__init__()
        pass

    def synthesize(self, specification: Specification, env: gym.Env) -> PlanStrategy:
        """Synthesizes a PlanStrategy in an env given the specification.

        Args:
            specification (Specification): _description_
            env (gym.Env): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Strategy: _description_
        """
        raise NotImplementedError()


class RLynthesisAlgorithm(SynthesisAlgorithm):
    """Reinforcement Algorithm based synthesis algorithm"""

    def __init__(self, rlalgorithm) -> None:
        super().__init__()
        self.rlalgorithm: Any = rlalgorithm

    def synthesize(self, specification: Specification, env: gym.Env) -> PolicyStrategy:
        """Synthesizes a PolicyStrategy in an env given the specification.

        Args:
            specification (Specification): _description_
            env (gym.Env): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Strategy: _description_
        """
        raise NotImplementedError()
