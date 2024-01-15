"""
>> from specless.system.tsbuilder import TSBuilder
>> from specless.system.tspbuilder import MiniGridSytemAndTSPAdapter
>> from specless.solver import MILPTSPSolver
>> import gym_minigrid # To load MiniGrid-BlockedUnlockPickup-v0

>> env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
>> tsbuilder = TSBuilder()
>> transition_system = tsbuilder(graph_data=env)
>> tsp = tspbuilder(transition_system)
>> tspsolver = MILPTSPSolver()
>> tours, costs = tspsolver(tsp)
>> tours
[[1,2,3,4,5]]
>> costs
[100]

>> env = MultiAgentWrapper(env, initial_states, concurrent=True)
>> tsbuilder = TSBuilder()
>> tspsolver = MILPTSPSolver()
>> tours, costs = tspsolver(tsp)
>> tours
[[1,2,3,4,5], [6,7,8,9,10], ..., [96,97,98,99,100]]
>> costs
[100, 120, ..., 90]
"""
import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
from bidict import bidict
from gymnasium.core import ActType

from specless.automaton.transition_system import MinigridTransitionSystem
from specless.specification.base import Specification
from specless.specification.timed_partial_order import TimedPartialOrder
from specless.tsp.tsp import GTSP, TSP, TSPWithTPO


class MiniGridSytemAndTSPAdapter:
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
        specification: Specification,
        initial_states=None,
        ignoring_obs_keys: List[str] = [],
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
            cost_ = []
            for tgt_node in nodes:
                shortest_path = self.all_pair_shortest_paths[src_node][tgt_node]
                if src_node == tgt_node:
                    cost = 0
                else:
                    cost = len(shortest_path)
                cost_.append(cost)
            costs.append(cost_)

        return GTSP(nodes, costs, nodesets=list(self.obs_to_nodes.values()))

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

    def map_back_to_controls(self, node_tour: List[int]) -> List[ActType]:
        ACTION_STR_TO_ENUM = {
            self.T.env.unwrapped.actions._member_names_[action]: action
            for action in self.T.env.unwrapped.actions
        }
        ACTION_ENUM_TO_STR = dict(
            zip(ACTION_STR_TO_ENUM.values(), ACTION_STR_TO_ENUM.keys())
        )

        controls = []

        node_edges = MiniGridSytemAndTSPAdapter.node_list_to_edges(node_tour)
        for src_node, tgt_node in node_edges:
            shortest_path = self.all_pair_shortest_paths[src_node][tgt_node]

            state_edges = MiniGridSytemAndTSPAdapter.node_list_to_edges(shortest_path)

            for src_state, tgt_state in state_edges:
                edge_data = self.T._get_edge_data(src_state, tgt_state)
                symbol = edge_data[0]["symbol"]
                controls.append(ACTION_STR_TO_ENUM[symbol])

        return controls

    @staticmethod
    def node_list_to_edges(nodes: List[int]) -> List[Tuple[int, int]]:
        edges = []
        for i in range(len(nodes) - 1):
            edge = (nodes[i], nodes[i + 1])
            edges.append(edge)

        return edges

    def build_mappings_from_ts(self, ignoring_obs_keys: List[str]) -> None:
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


class MiniGridSytemAndTSPAdapterWithTPO(MiniGridSytemAndTSPAdapter):
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

        global_constraints: Dict[
            int, Tuple[float, float]
        ] = self.convert_global_constraints(tpo)
        local_constraints: Dict[
            Tuple[int, int], Tuple[float, float]
        ] = self.convert_local_constraints(tpo)
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
