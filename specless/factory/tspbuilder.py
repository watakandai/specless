"""
>>> from specless.automaton.transition_system import TSBuilder
>>> from specless.factory.tspbuilder import TSPBuilder
>>> from specless.tsp.solver.milp import MILPTSPSolver
>>> import gymnasium as gym
>>> import gym_minigrid # To load MiniGrid-Empty-5x5-v0
>>> from specless.wrapper.minigridwrapper import MiniGridTransitionSystemWrapper

>>> env = gym.make("MiniGrid-Empty-5x5-v0")
>>> env = MiniGridTransitionSystemWrapper(env, ignore_direction=True)
>>> tsbuilder = TSBuilder()
>>> transition_system = tsbuilder(graph_data=env)
>>> tspbuilder = TSPBuilder()
>>> tsp = tspbuilder(transition_system)
>>> tspsolver = MILPTSPSolver()
>>> tours, costs = tspsolver.solve(tsp) # doctest: +ELLIPSIS
Restricted license...
>>> tours
[[0, 1, 0]]
>>> costs
10.0
"""

import copy
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from bidict import bidict
from gymnasium.core import ActType

from specless.automaton.transition_system import MinigridTransitionSystem
from specless.factory.builder import Builder
from specless.specification.base import Specification
from specless.specification.timed_partial_order import Service, TimedPartialOrder
from specless.strategy import CombinedStrategy, PlanStrategy, Strategy
from specless.tsp.tsp import GTSP, TSP, TSPWithTPO


class TSPBuilder(Builder):
    """Converts Transition System To a TSP Problem"""

    def __init__(self) -> None:
        """
        We call
        - TS states as "states"
        - TS observations as "observations"
        - TSP nodes as "nodes" or "locations"
        """

        self.T: Optional[MinigridTransitionSystem] = None

        """TS States to Observations"""
        self.state_to_obs: Dict[Tuple, str] = {}

        """Observation to Multiple TS States"""
        self.obs_to_states: Dict[str, List[Tuple]] = defaultdict(lambda: [])

        """Observation to Multiple TSP Nodes"""
        self.obs_to_nodes: Dict[str, List[int]] = defaultdict(lambda: [])

        """TS States to TSP Nodes bidict"""
        self.state_to_node: Dict[Tuple, int] = bidict()

        """all pair shortest paths"""
        # nodes, nodes => path
        self.all_pair_shortest_paths: Dict[Dict, Dict[int, List]] = defaultdict(
            lambda: defaultdict(lambda: [])
        )

    def __call__(
        self,
        T: MinigridTransitionSystem,
        specification: Optional[Specification] = None,
        initial_states=None,
        ignoring_obs_keys: List[str] = [],
        uniquelabel: bool = True,
    ) -> TSP:
        """_summary_

        Returns:
            _type_: _description_
        """
        self.T: MinigridTransitionSystem = T

        if initial_states is None:
            self.add_initial_state()
        else:
            for initial_state in initial_states:
                self.add_initial_state(initial_state)
        self.build_mappings_from_ts(ignoring_obs_keys)

        self.all_pair_shortest_paths = self.get_all_pair_shortest_paths()

        if specification is None:
            all_obs_exist = True
        else:
            must_visit_observations = list(specification.nodes())
            all_obs_exist = all(
                [obs in self.obs_to_nodes for obs in must_visit_observations]
            )

        if not all_obs_exist:
            env_name = T.env.unwrapped.spec.id
            raise Exception(
                f"Some of the observations cannot be observed in Env:{env_name}"
            )

        nodes = list(self.state_to_node.values())
        costs: List[List[float]] = []
        for src_node in nodes:
            cost_: List[float] = []
            for tgt_node in nodes:
                shortest_path = self.all_pair_shortest_paths[src_node][tgt_node]
                if src_node == tgt_node:
                    cost = 0
                else:
                    cost = len(shortest_path)
                cost_.append(cost)
            costs.append(cost_)

        if uniquelabel:
            return GTSP(nodes, costs, nodesets=list(self.obs_to_nodes.values()))
        return GTSP(nodes, costs)

    def get_all_pair_shortest_paths(self, weight_key: str = "weight"):
        G = copy.deepcopy(self.T)
        for src, tgt in G.edges():
            G[src][tgt][0][weight_key] = 1

        states = list(self.state_to_obs.keys())

        all_pair_shortest_paths = defaultdict(lambda: defaultdict(lambda: []))
        for src in states:
            for tgt in states:
                src_node = self.state_to_node[src]
                tgt_node = self.state_to_node[tgt]

                if src == tgt:
                    all_pair_shortest_paths[src_node][tgt_node] = []
                else:
                    try:
                        # Set edge weight to other visiting states very large so that it avoids the state
                        other_states = list(set(states) - {src, tgt})
                        for state in other_states:
                            for p in G.predecessors(state):
                                if p in [src, tgt]:
                                    continue
                                G[p][state][0][weight_key] = 100000

                        # There's a chance that there's no path between src & tgt.
                        distance, path = nx.single_source_dijkstra(
                            G, source=src, target=tgt, weight=weight_key
                        )
                        all_pair_shortest_paths[src_node][tgt_node] = path

                        for state in other_states:
                            for p in G.predecessors(state):
                                if p in [src, tgt]:
                                    continue
                                G[p][state][0][weight_key] = 1

                    except Exception:
                        # Set the edge weights back
                        other_states = list(set(states) - {src, tgt})
                        for state in other_states:
                            for p in G.predecessors(state):
                                if p in [src, tgt]:
                                    continue
                                G[p][state][0][weight_key] = 1

                        # TODO: For temporary, set it to a big number
                        all_pair_shortest_paths[src_node][tgt_node] = [""] * 100000
                        # raise e

        return all_pair_shortest_paths

    # TODO: Find the most highest (recent) layer wrapper's action space.
    def map_back_to_controls(self, node_tour: List[int]) -> List[ActType]:
        # TODO: Find the most highest (recent) layer wrapper's action space.
        actions = self.T.env.unwrapped.actions
        env = self.T.env
        while hasattr(env, "env"):
            if hasattr(env, "actions"):
                actions = env.actions
                break
            env = env.env

        ACTION_STR_TO_ENUM = {
            actions._member_names_[action]: action for action in actions
        }
        ACTION_ENUM_TO_STR = dict(
            zip(ACTION_STR_TO_ENUM.values(), ACTION_STR_TO_ENUM.keys())
        )

        controls = []

        node_edges = TSPBuilder.node_list_to_edges(node_tour)
        for src_node, tgt_node in node_edges:
            shortest_path = self.all_pair_shortest_paths[src_node][tgt_node]

            state_edges = TSPBuilder.node_list_to_edges(shortest_path)

            for src_state, tgt_state in state_edges:
                edge_data = self.T._get_edge_data(src_state, tgt_state)
                symbol = edge_data[0]["symbol"]
                controls.append(ACTION_STR_TO_ENUM[symbol])

        return controls

    def synthesize_strategy(self, tours) -> Strategy:
        actions: List[ActType] = [self.map_back_to_controls(tour) for tour in tours]

        if len(actions) == 0:
            assert False

        # Tours -> Strategy
        if len(actions) == 1:
            strategy = PlanStrategy(actions[0])
        else:
            strategy = CombinedStrategy([PlanStrategy(action) for action in actions])
        return strategy

    @staticmethod
    def node_list_to_edges(nodes: List[int]) -> List[Tuple[int, int]]:
        edges = []
        for i in range(len(nodes) - 1):
            edge = (nodes[i], nodes[i + 1])
            edges.append(edge)

        return edges

    def build_mappings_from_ts(
        self,
        ignoring_obs_keys: List[str],
    ) -> None:
        for state in self.T.nodes:
            obs = self.T.observe(state)
            if any([o in obs for o in ignoring_obs_keys]):
                continue

            if obs == "":
                continue

            self.state_to_obs[state] = obs

            if state not in self.state_to_node:
                node_id = len(self.state_to_node)
                self.state_to_node[state] = node_id
            else:
                node_id = self.state_to_node[state]

            self.obs_to_states[obs].append(state)
            self.obs_to_nodes[obs].append(node_id)

    def add_initial_state(self, initial_state: Tuple = None) -> None:
        if initial_state is None:
            initial_state = self.T.start_state

        i = sum("initial_state" in s for s in list(self.obs_to_states.keys()))

        obs = f"initial_state{i}"
        self.state_to_obs[initial_state] = obs
        node_id = len(self.state_to_node)
        self.state_to_node[initial_state] = node_id
        self.obs_to_states[obs].append(initial_state)
        self.obs_to_nodes[obs].append(node_id)


class TSPWithTPOBuilder(TSPBuilder):
    """Converts Transition System To a TSP Problem"""

    def __call__(
        self, T: MinigridTransitionSystem, specification: TimedPartialOrder, **kwargs
    ) -> TSPWithTPO:  # type: ignore[override]
        """Translate a MiniGrid Transition System to a TSP with

        Returns:
            _type_: _description_
        """
        tsp: TSP = super().__call__(T, specification, **kwargs)

        # MUST map TPO (w/ string labeled nodes) to TPO (with numbered nodes)
        numbered_tpo: TimedPartialOrder = self.convert_tpo(specification)

        return TSPWithTPO.from_tsp(tsp, numbered_tpo)

    def convert_tpo(self, tpo: TimedPartialOrder) -> TimedPartialOrder:
        """Convert the original TPO whose nodes are labeled with the observation names
        to a TPO with numbered nodes that correspond to the TSP problem.

        Args:
            tpo (TimedPartialOrder): _description_

        Returns:
            TimedPartialOrder: _description_
        """

        global_constraints: Dict[int, Tuple[float, float]] = (
            self.convert_global_constraints(tpo)
        )
        local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = (
            self.convert_local_constraints(tpo)
        )
        tpo = TimedPartialOrder.from_constraints(global_constraints, local_constraints)
        return tpo

    def convert_global_constraints(self, tpo: TimedPartialOrder) -> Dict:
        global_constraints: Dict[int, Tuple[float, float]] = {}
        for obs, bound in tpo.global_constraints.items():
            nodes = self.obs_to_nodes[obs]
            for n in nodes:
                global_constraints[n] = (bound["lb"], bound["ub"])

        return global_constraints

    def convert_local_constraints(self, tpo: TimedPartialOrder) -> Dict:
        local_constraints: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for src_obs, d in tpo.local_constraints.items():
            for tgt_obs, bound in d.items():
                src_nodes = self.obs_to_nodes[src_obs]
                tgt_nodes = self.obs_to_nodes[tgt_obs]

                for src_node in src_nodes:
                    for tgt_node in tgt_nodes:
                        local_constraints[(src_node, tgt_node)] = (
                            bound["lb"],
                            bound["ub"],
                        )

        return local_constraints


class AircraftTurnaroundTSPBuilder(Builder):
    """Build a TSP Problem from Aircraft Turnaround Minigrid"""

    def __init__(
        self,
        T,
        initial_locations,
        services,
        position_label_to_location,
        ignoring_obs_keys=[],
    ):
        """IMPORTANT: We call
        - TS states as "states"
        - Service (df 'Activity) as "service"
        - TSP nodes as "nodes"
        """
        self.T = T
        # Add initial state "Services"
        initial_services = [
            Service(f"Robot{i}Start", "Stay", f"Init{i}", f"Init{i}", 0, True, [])
            for i, l in enumerate(initial_locations)
        ]
        self.services = services + initial_services
        self.position_label_to_location = position_label_to_location | {
            f"Init{i}": l for i, l in enumerate(initial_locations)
        }
        self.ignoring_obs_keys = ignoring_obs_keys

        """Service to TSP Node"""
        self.service_to_node = bidict({s: i for i, s in enumerate(self.services)})
        self.service_name_to_node = bidict(
            {s.name: i for i, s in enumerate(self.services)}
        )

        """Uncontrollable"""
        self.uncontrollables = [s for s in services if not s.controllable]

        """TS Nodes Service Time"""
        self.service_time = defaultdict(lambda: 0)
        self.service_path = defaultdict(lambda: [])

        """all pair shortest paths"""
        # nodes, nodes => path
        self.all_pair_shortest_paths = defaultdict(lambda: defaultdict(lambda: []))
        """Between States"""
        self.all_pair_state_shortest_paths = defaultdict(
            lambda: defaultdict(lambda: [])
        )

        """For convenience"""
        self.name_to_service = {s.name: s for s in self.services}

    def __call__(
        self,
        T: MinigridTransitionSystem,
        specification: TimedPartialOrder,
        initial_states=None,
        ignoring_obs_keys: List[str] = [],
        uniquelabel: bool = True,
        **kwargs,
    ) -> TSPWithTPO:  # type: ignore[override]
        """Translate a MiniGrid Transition System to a TSP with

        Returns:
            _type_: _description_
        """
        self.T: MinigridTransitionSystem = T

        if initial_states is None:
            self.add_initial_state()
        else:
            for initial_state in initial_states:
                self.add_initial_state(initial_state)
        self.build_mappings_from_ts(ignoring_obs_keys)
        self.all_pair_shortest_paths = self.get_all_pair_shortest_paths()

        # MUST map TPO (w/ string labeled nodes) to TPO (with numbered nodes)
        numbered_tpo: TimedPartialOrder = self.convert_tpo(specification)

        return TSPWithTPO.from_tsp(tsp, numbered_tpo)

    def location_to_state(self, location):
        return ", ".join([str(location), "right"])

    def state_to_location(self, state):
        m = re.match(r"\(([\d]*), ([\d]*)\), ([a-z]*)", state)
        return (int(m.group(1)), int(m.group(2)))

    def to_tsp_nodes_and_costs(self):
        self.get_all_pair_state_shortest_paths()
        costs, service_costs = self.get_all_pair_shortest_paths()
        nodes = list(costs.keys())
        costs = [[c for tgt, c in d.items()] for src, d in costs.items()]
        return nodes, costs, service_costs

    def get_all_pair_state_shortest_paths(self, weight_key: str = "weight"):
        # TODO: Set 1 if NESW and set 1.41 if NE,SE,SW,NW
        G = copy.deepcopy(self.T)
        for src, tgt in G.edges():
            src_loc = np.array(self.state_to_location(src))
            tgt_loc = np.array(self.state_to_location(tgt))
            # Diagnoal Move!
            if np.linalg.norm(src_loc - tgt_loc) > 1:
                # G[src][tgt][0][weight_key] = 1.4 * 0.51
                # G[src][tgt][0][weight_key] = 1 * 0.5
                G[src][tgt][0][weight_key] = 2
            # Else North, South, East, West
            else:
                # G[src][tgt][0][weight_key] = 1 * 0.5
                G[src][tgt][0][weight_key] = 1

        to_states = [
            self.location_to_state(self.position_label_to_location[s.to_str])
            for s in self.services
            if s not in self.uncontrollables
        ]
        from_states = [
            self.location_to_state(self.position_label_to_location[s.from_str])
            for s in self.services
            if s not in self.uncontrollables
        ]

        states = set(to_states + from_states)

        # Compute d = |l1 - l2|^1
        for src in states:
            for tgt in states:
                if src == tgt:
                    self.all_pair_state_shortest_paths[src][tgt] = (0, [])
                else:
                    try:
                        # Set edge weight to other visiting states very large so that it avoids the state
                        other_states = list(set(states) - {src, tgt})
                        for state in other_states:
                            for p in G.predecessors(state):
                                G[p][state][0][weight_key] = 100000

                        distance, path = nx.single_source_dijkstra(
                            G, source=src, target=tgt, weight=weight_key
                        )
                        self.all_pair_state_shortest_paths[src][tgt] = (distance, path)

                        # Set the edge weights back
                        other_states = list(set(states) - {src, tgt})
                        for state in other_states:
                            for p in G.predecessors(state):
                                G[p][state][0][weight_key] = 1
                    except Exception as e:
                        print(src, tgt)
                        raise e

    def get_all_pair_shortest_paths(self, weight_key: str = "weight"):
        """Must Compute the Cost Between Services & Service Cost.

        Args:
            weight_key (str, optional): _description_. Defaults to 'weight'.
        """
        costs = defaultdict(lambda: defaultdict(lambda: 0))
        service_costs = defaultdict(lambda: 0)

        for service, node in self.service_to_node.items():
            if service in self.uncontrollables:
                service_costs[node] = service.service_time
                continue

            state_from = self.location_to_state(
                self.position_label_to_location[service.from_str]
            )
            state_to = self.location_to_state(
                self.position_label_to_location[service.to_str]
            )

            if state_from == state_to:
                service_costs[node] = service.service_time
            else:
                distance, _ = self.all_pair_state_shortest_paths[state_from][state_to]
                service_costs[node] = distance

        for src_service, src_node in self.service_to_node.items():
            if src_service not in self.uncontrollables:
                src_state_to = self.service_position_label_to_state(src_service.to_str)

            for tgt_service, tgt_node in self.service_to_node.items():
                if (
                    src_service in self.uncontrollables
                    or tgt_service in self.uncontrollables
                ):
                    costs[src_node][tgt_node] = 0
                    self.all_pair_shortest_paths[src_node][tgt_node] = (0, [])
                    continue

                tgt_state_from = self.service_position_label_to_state(
                    tgt_service.from_str
                )

                if src_state_to == tgt_state_from:
                    costs[src_node][tgt_node] = 0
                    self.all_pair_shortest_paths[src_node][tgt_node] = (0, [])
                else:
                    distance, path = self.all_pair_state_shortest_paths[src_state_to][
                        tgt_state_from
                    ]
                    costs[src_node][tgt_node] = distance
                    self.all_pair_shortest_paths[src_node][tgt_node] = (distance, path)

        return costs, service_costs

    def service_position_label_to_state(self, position_label):
        return self.location_to_state(self.position_label_to_location[position_label])

    def to_controls(self, node_tour, times, return_states_as_waypoints: bool = False):
        controls = []
        trace = []
        if len(node_tour) == 0:
            return controls, trace

        node_edges = TSConverter.node_list_to_edges(node_tour)
        init_service = self.service_to_node.inverse[node_tour[0]]
        if init_service in self.uncontrollables:
            return controls, trace
        trace.append(node_tour[0])

        curr_state = self.service_position_label_to_state(init_service.from_str)
        curr_step = 0

        for src_node, tgt_node in node_edges:
            # src_service = self.service_to_node.inverse[src_node]
            # tgt_service = self.service_to_node.inverse[tgt_node]
            # Must take 2 steps
            # 1.Travel: src_node.to -> tgt_node.from
            # 2. Service: tgt_node.from -> tgt_node.to

            # Travel: src_node.to -> tgt_node.from
            _, shortest_path = self.all_pair_shortest_paths[src_node][tgt_node]
            state_edges = TSConverter.node_list_to_edges(shortest_path)

            for src_state, tgt_state in state_edges:
                edge_data = self.T._get_edge_data(src_state, tgt_state)
                symbol = edge_data[0]["symbol"]
                controls.append(symbol)
                trace.append(-1)
                curr_state = tgt_state
                curr_step += 1

            # Service: tgt_node.from -> tgt_node.to
            service = self.service_to_node.inverse[tgt_node]
            from_state = self.service_position_label_to_state(service.from_str)
            to_state = self.service_position_label_to_state(service.to_str)

            # Important: Wait until the precedent services are completed
            # We cannot start the service until other precedent tasks are done!!!
            prec_ends = [
                times["end"][self.service_to_node[self.name_to_service[p]]]
                for p in service.precedences
            ]
            maximum_prec_end_time = 0 if len(prec_ends) == 0 else max(prec_ends)
            while curr_step < maximum_prec_end_time:
                edge_data = self.T._get_edge_data(curr_state, curr_state)
                symbol = edge_data[0]["symbol"]
                controls.append(symbol)
                trace.append(-1)
                curr_step += 1

            _, shortest_path = self.all_pair_state_shortest_paths[from_state][to_state]
            # Service at the station
            if len(shortest_path) == 0 and not return_states_as_waypoints:
                duration = service.service_time
                edge_data = self.T._get_edge_data(curr_state, curr_state)
                symbol = edge_data[0]["symbol"]
                controls += [symbol] * duration
                trace += [-1] * duration
                curr_step + duration
            state_edges = TSConverter.node_list_to_edges(shortest_path)
            for src_state, tgt_state in state_edges:
                edge_data = self.T._get_edge_data(src_state, tgt_state)
                symbol = edge_data[0]["symbol"]
                controls.append(symbol)
                trace.append(-1)
                curr_state = tgt_state
                curr_step += 1

            trace[-1] = tgt_node

        trace[-1] = -1
        return controls, trace

    def tour_to_services(self, tour):
        return [self.service_to_node.inverse[n] for n in tour]
