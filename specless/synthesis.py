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
>>> import os
>>> import gymnasium as gym
>>> import specless as sl
>>> from specless.minigrid.tspenv import TSPEnv  # NOQA

>>> env = gym.make("MiniGrid-TSP-v0", render_mode="rgb_array", seed=3)
>>> env = sl.MiniGridTransitionSystemWrapper(
...     env, ignore_direction=True, skip_observations=["unseen", "wall"]
... )
>>> pdfabuilder = sl.PDFABuilder()
>>> specification = pdfabuilder(os.path.join(os.getcwd(), "examples/demo/pdfa.yaml"))
>>> synthesis_algorithm = sl.ProductGraphSynthesisAlgorithm()
>>> strategy = synthesis_algorithm.synthesize(env, specification)
"""

from abc import ABCMeta
from typing import Any, List, Optional

import gymnasium as gym

from specless.automaton.pdfa import PDFA
from specless.automaton.product import ProductBuilder
from specless.automaton.transition_system import MinigridTransitionSystem, TSBuilder
from specless.factory.tspbuilder import (
    AircraftTurnaroundTSPBuilder,
    TSPBuilder,
    TSPWithTPOBuilder,
)
from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.strategy import (
    PlanStrategy,
    Strategy,
)
from specless.tsp.solver.milp import MILPTSPSolver, MILPTSPWithTPOSolver
from specless.tsp.tsp import Node, TSPWithTPO
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
        init_nodes: Optional[List[Node]] = None,
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
            tours, cost, timestamps = tspsolver.solve(
                tsp_with_tpo, num_agent=num_agent, init_nodes=init_nodes
            )
        else:
            # Convert the Transition System to a Traveling Saleseman Problem
            tspbuilder = TSPBuilder()
            # 2: Create a Specification Class with just a list of nodes
            tsp = tspbuilder(transition_system, uniquelabel=False)

            # Solve the TSP and obtain tours
            tspsolver = MILPTSPSolver()
            tours, cost = tspsolver.solve(
                tsp, num_agent=num_agent, init_nodes=init_nodes
            )

        return tspbuilder.synthesize_strategy(tours)


class ServiceTSPSynthesisAlgorithm(SynthesisAlgorithm):
    """Traveling Salesman Problem based synthesis algorithm"""

    def __init__(self, tspsolver=None) -> None:
        super().__init__()

    def synthesize(
        self,
        env: gym.Env,
        specification: Specification,
        num_agent: int = 1,
        init_nodes: Optional[List[Node]] = None,
        services=None,
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

        # =========================================
        tspbuilder = AircraftTurnaroundTSPBuilder(
            transition_system,
            init_nodes,
            services,
            ignoring_obs_keys=["empty", "lava"],
        )
        nodes, costs, service_times = tspbuilder.to_tsp_nodes_and_costs()
        # =========================================

        global_constraints = {}
        local_constraints = self.construct_tpo_constraints(specification, tspbuilder)

        # =========================================
        tpo = TPO(global_constraints, local_constraints)
        tsptc = TSPTC(nodes, costs, tpo, service_times)
        tsp_with_tpo: TSPWithTPO = TSPWithTPO(tsptc)

        # =========================================
        init_nodes = [
            n for n in nodes if "Robot" in tspbuilder.service_to_node.inverse[n].name
        ]
        uncontrollable_init_nodes = [
            tspbuilder.service_to_node[s]
            for s in tspbuilder.uncontrollables
            if s.name == "Arrival"
        ]
        init_nodes.append(uncontrollable_init_nodes[0])

        # =========================================
        solver = MILPTSPWithTPOSolver(tsp_with_tpo, init_nodes=init_nodes)
        # TPO & TransitionSystem -> TSP
        tspbuilder = TSPWithTPOBuilder()
        tsp_with_tpo: TSPWithTPO = tspbuilder(transition_system, specification)

        # Solve TSP -> Tours
        tspsolver = MILPTSPWithTPOSolver()
        # TODO: tsp argument should be passed to the solve() function
        tours, cost, timestamps = tspsolver.solve(
            tsp_with_tpo, num_agent=num_agent, init_nodes=init_nodes
        )

        return tspbuilder.synthesize_strategy(tours)

    def construct_tpo_constraints(specification, ts_converter):
        local_constraints = {}
        for src, d in specification.nodes(data=True):
            src_service_name = d["Activity"]
            src_node = ts_converter.service_name_to_node[src_service_name]
            for tgt in specification.successors(src):
                tgt_service_name = specification.nodes[tgt]["Activity"]
                tgt_node = ts_converter.service_name_to_node[tgt_service_name]
                service = ts_converter.name_to_service[tgt_service_name]
                lb = 0
                if service.service_time > 0:
                    lb = service.service_time
                local_constraints[(src_node, tgt_node)] = (lb, float("inf"))
            return local_constraints


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
