from abc import ABCMeta
from typing import Any, List

import gymnasium as gym
from gymnasium.core import ActType

from specless.automaton.transition_system import MinigridTransitionSystem, TSBuilder
from specless.factory.tspadapter import MiniGridSytemAndTSPAdapterWithTPO
from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.strategy import (
    CombinedStrategy,
    PlanStrategy,
    Strategy,
)
from specless.tsp.solver.milp import MILPTSPWithTPOSolver
from specless.tsp.tsp import TSPWithTPO
from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper


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

    def synthesize(self, specification: Specification, env: gym.Env) -> Strategy:
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

    def __init__(self, tspsolver=None) -> None:
        super().__init__()

    def synthesize(self, specification: TimedPartialOrder, env: gym.Env) -> Strategy:
        """Synthesizes a PlanStrategy in an env given the specification.

        Env+#Agent -> TS+#Agent -> (Nodes, Costs, ServiceTimes)+#Agent -> List[Strategy]
        -> MultiEnv
        * How should I deal with multiple robots?

        Args:
            specification (Specification): _description_
            env (gym.Env): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Strategy: _description_
        """
        # Env -> TransitionSystem
        env = MiniGridTransitionSystemWrapper(env)
        tsbuilder = TSBuilder()
        transition_system: MinigridTransitionSystem = tsbuilder(
            env, graph_data_format="minigrid"
        )
        # transition_system.draw("MiniGrid-Empty-5x5-v0")

        # TPO & TransitionSystem -> TSP
        adapter = MiniGridSytemAndTSPAdapterWithTPO()
        tsp_with_tpo: TSPWithTPO = adapter(transition_system, specification)

        # Solve TSP -> Tours
        tspsolver = MILPTSPWithTPOSolver()
        # TODO: tsp argument should be passed to the solve() function
        tours, cost = tspsolver.solve(tsp_with_tpo)

        # TODO: Convert tours to a sequence of actions...
        actions: List[ActType] = [adapter.map_back_to_controls(tour) for tour in tours]

        if len(actions) == 0:
            assert False

        # Tours -> Strategy
        if len(actions) == 1:
            strategy = PlanStrategy(actions[0])
        else:
            strategy = CombinedStrategy([PlanStrategy(action) for action in actions])
        return strategy


class RLynthesisAlgorithm(SynthesisAlgorithm):
    """Reinforcement Algorithm based synthesis algorithm"""

    def __init__(self, rlalgorithm) -> None:
        super().__init__()
        self.rlalgorithm: Any = rlalgorithm

    def synthesize(self, specification: Specification, env: gym.Env) -> Strategy:
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
