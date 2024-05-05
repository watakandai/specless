"""
====================
Synthesis Algorithms
====================

Synthesis module contains classes and functions for synthesizing strategies from specifications.

This module contains the following classes:
    * SynthesisAlgorithm: Base class for all synthesis algorithms.
    * ProductGraphSynthesisAlgorithm: Product Graph based synthesis algorithm.
    * TSPSynthesisAlgorithm: Traveling Salesman Problem based synthesis algorithm.
    * RLynthesisAlgorithm: Reinforcement Learning based synthesis algorithm.

Examples
--------
>>> from specless.synthesis import ProductGraphSynthesisAlgorithm
>>> from specless.automaton.pdfa import PDFA
>>> from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper
>>> from specless.specification.base import Specification
>>> from specless.strategy import Strategy
>>> env = MiniGridTransitionSystemWrapper()
>>> specification = PDFA()
>>> synthesis_algorithm = ProductGraphSynthesisAlgorithm()
>>> strategy: Strategy = synthesis_algorithm.synthesize(env, specification)
"""

from abc import ABCMeta
from typing import Any

import gymnasium as gym

from specless.automaton.pdfa import PDFA
from specless.automaton.product import ProductBuilder
from specless.automaton.transition_system import MinigridTransitionSystem, TSBuilder
from specless.factory.tspbuilder import TSPBuilder, TSPWithTPOBuilder
from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.strategy import (
    PlanStrategy,
    Strategy,
)
from specless.tsp.solver.milp import MILPTSPSolver, MILPTSPWithTPOSolver
from specless.tsp.tsp import TSPWithTPO
from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper


class SynthesisAlgorithm(metaclass=ABCMeta):
    """Base classs for all synthesis algorithms"""

    def __init__(self) -> None:
        pass

    def synthesize(
        self, env: gym.Env, specification: Specification, *args, **kwargs
    ) -> Strategy:
        """Synthesizes a strategy in an env given the specification.

        Parameters
        ----------
        env : gym.Env
            The environment in which to synthesize the strategy.
        specification : Specification
            The specification for the strategy.

        Returns
        -------
        Strategy
            The synthesized strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


class ProductGraphSynthesisAlgorithm(SynthesisAlgorithm):
    """Product Graph based synthesis algorithm"""

    def __init__(self) -> None:
        super().__init__()
        pass

    # TODO: String action to Enum action
    def synthesize(
        self, env: gym.Env, specification: Specification, *args, **kwargs
    ) -> Strategy:
        """Synthesizes a MemorylessStrategy in an env given the specification.

        Synthesize a MemorylessStrategy in an environment given the specification.

        Parameters
        ----------
        env : gym.Env
            The environment in which to synthesize the strategy.
        specification : Specification
            The specification for the strategy.

        Returns
        -------
        Strategy
            The synthesized strategy.

        Raises
        ------
        Exception
            If the environment is not wrapped by MiniGridTransitionSystemWrapper or
            if the specification is not of type DFA or PDFA.
        """
        # Env -> TransitionSystem
        if not isinstance(env, MiniGridTransitionSystemWrapper):
            raise Exception("env must be wrapped by MiniGridTransitionSystemWrapper")

        if not isinstance(specification, PDFA):
            raise Exception("Specification must be of type DFA or PDFA")

        tsbuilder = TSBuilder()
        transition_system: MinigridTransitionSystem = tsbuilder(env)

        productbuilder = ProductBuilder()
        product_graph = productbuilder(graph_data=(transition_system, specification))

        controls_symbols, obs_prob = product_graph.compute_strategy()
        return PlanStrategy(controls_symbols)


class TSPSynthesisAlgorithm(SynthesisAlgorithm):
    """Traveling Salesman Problem based synthesis algorithm"""

    def __init__(self, tspsolver=None) -> None:
        super().__init__()

    def synthesize(
        self,
        env: gym.Env,
        specification: Specification,
        num_agent: int = 1,
        *args,
        **kwargs,
    ) -> Strategy:
        """Synthesizes a PlanStrategy in an env given the specification.

        Env+#Agent -> TS+#Agent -> (Nodes, Costs, ServiceTimes)+#Agent -> List[Strategy]
        -> MultiEnv
        * How should I deal with multiple robots?

        Parameters
        ----------
        env : gym.Env
            The environment in which to synthesize the strategy.
        specification : Specification
            The specification for the strategy.
        num_agent : int, optional
            The number of agents. Default is 1.

        Returns
        -------
        Strategy
            The synthesized strategy.

        Raises
        ------
        Exception
            If the environment is not wrapped by MiniGridTransitionSystemWrapper.
        """
        # Env -> TransitionSystem
        if not isinstance(env, MiniGridTransitionSystemWrapper):
            print(type(env))
            raise Exception("env must be wrapped by MiniGridTransitionSystemWrapper")

        tsbuilder = TSBuilder()
        transition_system: MinigridTransitionSystem = tsbuilder(env)

        if isinstance(specification, TimedPartialOrder):
            # TPO & TransitionSystem -> TSP
            tspbuilder = TSPWithTPOBuilder()
            tsp_with_tpo: TSPWithTPO = tspbuilder(transition_system, specification)

            # Solve TSP -> Tours
            tspsolver = MILPTSPWithTPOSolver()
            # TODO: tsp argument should be passed to the solve() function
            tours, cost, timestamps = tspsolver.solve(tsp_with_tpo, num_agent=num_agent)
        else:
            # Convert the Transition System to a Traveling Saleseman Problem
            tspbuilder = TSPBuilder()
            # 2: Create a Specification Class with just a list of nodes
            tsp = tspbuilder(transition_system, uniquelabel=False)

            # Solve the TSP and obtain tours
            tspsolver = MILPTSPSolver()
            tours, cost = tspsolver.solve(tsp, num_agent=3)

        return tspbuilder.synthesize_strategy(tours)


class RLynthesisAlgorithm(SynthesisAlgorithm):
    """Reinforcement Algorithm based synthesis algorithm"""

    def __init__(self, rlalgorithm) -> None:
        super().__init__()
        self.rlalgorithm: Any = rlalgorithm

    def synthesize(self, env: gym.Env, specification: Specification) -> Strategy:
        """Synthesizes a PolicyStrategy in an env given the specification.

        Parameters
        ----------
        env : gym.Env
            The environment in which to synthesize the strategy.
        specification : Specification
            The specification for the strategy.

        Returns
        -------
        Strategy
            The synthesized strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()
