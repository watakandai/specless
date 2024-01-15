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
from collections.abc import Iterable
from typing import Dict, List, Tuple, Union

from gym_minigrid.minigrid import MiniGridEnv
from gymnasium.core import ActType

from specless.const import (
    IDX_TO_COLOR,
    MINIGRID_TO_GRAPHVIZ_COLOR,
)
from specless.typing import ActionsEnum
from specless.wrapper.labelwrapper import (
    AddPosDirToMiniGridWrapper,
    LabelMiniGridWrapper,
)
from specless.wrapper.tswrapper import TransitionSystemWrapper


class MiniGridTransitionSystemWrapper(TransitionSystemWrapper):
    """_summary_

    MiniGridEnv returns a state of type Dict
    state: Dict = {
        'image': image,
        'direction': self.agent_dir,
        'mission': self.mission
    }

    Args:
        TransitionSystemWrapper (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    LABELKEY = "observation"

    def __init__(
        self,
        env: MiniGridEnv,
        skip_observations: List[str] = ["unseen", "wall", "empty"],
        ignore_done: bool = True,
    ):
        # Label each state using LabelMiniGridWrapper
        env = LabelMiniGridWrapper(
            env,
            labelkey=self.LABELKEY,
            skiplist=skip_observations,
        )
        env = AddPosDirToMiniGridWrapper(env)
        super().__init__(env, ignore_done=ignore_done)
        """_summary_

        MiniGridEnv
        =============
        # Action enumeration for this environment
        self.actions: IntEnum = MiniGridEnv.Actions
        # Actions are discrete integer values
        self.action_space: gym.spaces.Discrete = spaces.Discrete(len(self.actions))
        """

        # building some more constant DICTS dynamically from the env data
        ACTION_STR_TO_ENUM = {
            self.unwrapped.actions._member_names_[action]: action
            for action in self.actions()
        }
        self.ACTION_ENUM_TO_STR = dict(
            zip(ACTION_STR_TO_ENUM.values(), ACTION_STR_TO_ENUM.keys())
        )

    def actions(self) -> Union[Iterable, ActionsEnum]:
        return self.unwrapped.actions

    def _get_action_str(self, action: ActType) -> str:
        return self.ACTION_ENUM_TO_STR[action]

    def _get_node_from_state(self, state: Dict) -> Tuple:
        """Get current "node" from state.
        We only need agent's position and direction.

        Args:
            state (Dict): _description_

        Returns:
            Tuple: _description_
        """
        # Copied from gym_minigrid.minigrid.MiniGridEnv.__str__
        AGENT_DIR_TO_STR: dict[int, str] = {0: ">", 1: "V", 2: "<", 3: "^"}
        return state["pos"], AGENT_DIR_TO_STR[state["dir"]]

    def _get_state(self, obs) -> Dict:
        """Returns the current state.
        AUGMENT ANY NECESSARY INFO that is required in other functions

        MiniGrid Env Observations are dictionaries containing:
            - an image (partially observable view of the environment)
            - the agent's direction/orientation (acting as a compass)
            - a textual mission string (instructions for the agent)

        Additionally, we need the agent's position and the direction

        Returns:
            Dict: _description_
        """
        return obs

    def _set_state(self, state) -> None:
        base_env = self.unwrapped

        position, direction = state["pos"], state["dir"]

        if position is not None:
            base_env.agent_pos = position

        if direction is not None:
            base_env.agent_dir = direction

    def _add_state_data(self, state_data: Dict, state: Dict):
        base_env = self.env.unwrapped
        pos = base_env.agent_pos
        col_idx, row_idx = pos

        img = state["image"]
        obs = img[col_idx, row_idx]

        _, obj_color, _ = obs
        color = IDX_TO_COLOR[obj_color]

        color = MINIGRID_TO_GRAPHVIZ_COLOR[color]

        state_data.update({"color": color})
        return state_data

    def _get_obs_str_from_state(self, state: Dict) -> str:
        """Return an observation string from state
        The state is "augmented" by the LabelMinigridWrapper.
        It can be accessed via key LABELKEY specified when
        LabelMinigridWrapper was instantiated.

        Args:
            state (Dict): _description_

        Returns:
            _type_: _description_
        """
        return state[self.LABELKEY]
