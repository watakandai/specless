"""
SpeclessEnv
===========
A standard gym.Env is accepted if the states and actions are finite
(Discrete Obs and Action Space)
>> import gymnasium as gym
>> env = gym.make("CustomEnv-v0")
>> env.obs_space
Dict(Discrete(), Text())


Wrapper
=======
A standard gym environment with other spaces (e.g., Dict)
can be translated into a SpeclessEnv by providing the


>> from specless.gym.wrappers import SpeclessWwrapper
>> env = SpeclessWwrapper(env, states, actions)
* Note, continuous space will be supported in the future
(using Sampled-based planners to translate the env into a finite system.)

If wanted, we can extend it to multiple agents
>> from specless.gym.wrappers import MultiAgentWrapper
>> initial_states = [(1, 1), (2, 2), (3, 3)]
>> env = MultiAgentWrapper(env, initial_states, concurrent=False) # Turn-based

Transition System Builder
=========================
>> from specless.system import TSBuilder
>> env: SpeclessEnv = gym.make("CustomEnv-v0")
>> actions = env.action_space.start + np.arange(env.action_space.n)
>> tsbuilder = TSBuilder(actions)
>> ts = tsbuilder(env)

For multiple agents
>> env: SpeclessEnv = gym.make("CustomEnv-v0")
>> initial_states = [(1, 1), (2, 2), (3, 3)]
>> env = MultiAgentWrapper(env, initial_states, concurrent=True)
>> tsbuilder = TSBuilder()
>> ts = tsbuilder(env)

Users can set a function to label nodes
>> tsbuilder.set_add_node_func(add_node_func)

and a function to set edge labels
>> tsbuilder.set_add_edge_func(add_edge_func)
"""
import queue
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, Tuple, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType

# EnvType = Union[SpeclessEnv, SpeclessEnvWrapper]
EnvType = gym.Env
EnvObs = Dict
CellObs = Tuple[int, int, int]
ActionsEnum = Enum
Reward = float
Done = bool
StepData = Tuple[EnvObs, Reward, Done, dict]


class TransitionSystemWrapper(gym.core.Wrapper, metaclass=ABCMeta):
    """
    Wrapper to define an environment that can be represented as a transition
    system.
    """

    def __init__(self, env: EnvType, ignore_done: bool = False) -> None:
        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(env)
        self.reset()
        self.ignore_done = ignore_done

    @abstractmethod
    def actions(self) -> Union[Iterable, ActionsEnum]:
        raise NotImplementedError()

    @abstractmethod
    def _get_node_from_state(self, state: Dict) -> Tuple:
        raise NotImplementedError()

    @abstractmethod
    def _get_obs_str_from_state(self, state: Dict) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _get_state(self, obs: ObsType) -> Dict:
        raise NotImplementedError()

    @abstractmethod
    def _set_state(self, state) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _get_action_str(self, action: ActType) -> str:
        raise NotImplementedError()

    def _add_state_data(self, state_data: Dict, state: Dict):
        return state_data

    def _get_env_prop(self, env_property_name: str):
        """
        Gets the base environment's property.

        :param      env_property_name:  The base environment's property name

        :returns:   The base environment's property.
        """

        base_env = self.env.unwrapped

        return getattr(base_env, env_property_name)

    def _set_env_prop(self, env_property_name: str, env_property) -> None:
        """
        Sets the base environment's property.

        :param      env_property_name:  The base environment's property name
        :param      env_property:       The new base environment property data
        """

        base_env = self.env.unwrapped
        setattr(base_env, env_property_name, env_property)

    def _add_node(
        self, nodes: dict, state: Dict, terminated: bool
    ) -> Tuple[Dict, Tuple]:
        """
        Adds a node to the dict of nodes used to initialize an automaton obj.

        :param      nodes:             dict of nodes to build the automaton out
                                       of. Must be in the format needed by
                                       networkx.add_nodes_from()
        :param      pos:               The agent's position
        :param      direction:         The agent's direction
        :param      obs_str:           The state observation string

        :returns:   (updated dict of nodes, new label for the added node)
        """
        node = self._get_node_from_state(state)

        # Get an observation from the state
        obs_str = self._get_obs_str_from_state(state)

        if node not in nodes:
            state_data = {
                "trans_distribution": None,
                "observation": obs_str,
                "is_accepting": terminated,
            }
            nodes[node] = self._add_state_data(state_data, state)

        return nodes, node

    def _add_edge(
        self,
        nodes: dict,
        edges: dict,
        action: ActType,
        edge: Tuple[Dict, Dict],
        terminated: bool,
    ) -> Tuple[dict, dict]:
        """
        Adds both nodes to the dict of nodes and to the dict of edges used to
        initialize an automaton obj.

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()
        :param      action:              The action taken
        :param      edge:                The edge to add

        :returns:   (updated dict of nodes, updated dict of edges)
        """

        action_str: str = self._get_action_str(action)

        src_state, dest_state = edge
        nodes, src_node = self._add_node(nodes, src_state, False)
        nodes, dest_node = self._add_node(nodes, dest_state, terminated)

        edge_data: dict[str, list[str]] = {"symbols": [action_str]}
        edge_dict = {dest_node: edge_data}

        if src_node in edges:
            if dest_node in edges[src_node]:
                existing_edge_data = edges[src_node][dest_node]
                existing_edge_data["symbols"].extend(edge_data["symbols"])
                edges[src_node][dest_node] = existing_edge_data
            else:
                edges[src_node].update(edge_dict)
        else:
            edges[src_node] = edge_dict

        return nodes, edges

    def make_transition(
        self, src_state: Dict, action: ActType
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        self._set_state(src_state)
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)
        dest_state: Dict = self._get_state(obs)

        return dest_state, reward, terminated, truncated, info

    def extract_transition_system(self) -> dict:
        """
        Extracts all data needed to build a transition system representation of
        the environment.

        :returns:   The transition system data.
        """
        state: Dict
        nodes: dict = {}
        edges: dict = {}

        state, info = self.reset()

        q: queue.Queue = queue.Queue()
        q.put(state)

        # Cannot add an unhashable type: 'list'. So store a node (Tuple)
        visited = set()
        # Cannot add an unhashable type: 'list'. So store a node (Tuple)
        done_states = set()

        while not q.empty():
            src_state: Dict = q.get()
            src_node: Tuple = self._get_node_from_state(src_state)
            visited.add(src_node)

            for action in self.actions():
                if src_node not in done_states:
                    (
                        dest_state,
                        _,
                        terminated,
                        truncated,
                        _,
                    ) = self.make_transition(src_state, action)

                    dest_node: Tuple = self._get_node_from_state(dest_state)

                    possible_edge: Tuple[Dict, Dict] = (src_state, dest_state)

                    nodes, edges = self._add_edge(
                        nodes, edges, action, possible_edge, terminated
                    )

                    # don't want to add outgoing transitions from states that
                    # we know are done to the TS, as these are wasted space
                    if not self.ignore_done and terminated or truncated:
                        done_states.add(dest_node)
                        # need to reset after done, to clear the 'done' state
                        self.reset()

                    if dest_node not in visited:
                        q.put(dest_state)
                        visited.add(dest_node)

        # we have moved the agent a bunch, so we should reset it when done
        # extracting all of the data.
        self.reset()

        return self._package_data(nodes, edges)

    def _package_data(self, nodes: dict, edges: dict) -> dict:
        """
        Packages up extracted data from the environment in the format needed by
        automaton constructors

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()

        :returns:   configuration data dictionary
        """

        config_data: Dict[str, Any] = {}

        # can directly compute these from the graph data
        symbols = set()
        state_labels = set()
        observations = set()
        for state, edge in edges.items():
            for _, edge_data in edge.items():
                symbols.update(edge_data["symbols"])
                state_labels.add(state)

        for node in nodes.keys():
            observation = nodes[node]["observation"]
            observations.add(observation)

        alphabet_size: int = len(symbols)
        num_states: int = len(state_labels)
        num_obs: int = len(observations)

        # TODO: Get the initial state
        state, _ = self.reset()
        start_node: str = self._get_node_from_state(state)

        config_data["alphabet_size"] = alphabet_size
        config_data["num_states"] = num_states
        config_data["num_obs"] = num_obs
        config_data["nodes"] = nodes
        config_data["edges"] = edges
        config_data["start_state"] = start_node

        return config_data
