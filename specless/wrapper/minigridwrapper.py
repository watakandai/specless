"""
SpeclessEnv
===========
A standard gym.Env is accepted if the states and actions are finite
(Discrete Obs and Action Space)
>>> import gymnasium as gym
>>> from specless.minigrid.tspenv import TSPEnv  # NOQA
>>> env = gym.make("MiniGrid-TSP-v0")

# >>> env.obs_space
# Dict(Discrete(), Text())


Wrapper
=======
A standard gym environment with other spaces (e.g., Dict)
can be translated into a SpeclessEnv by providing the

# TODO: >>>
# >>> from specless.minigrid.core import SpeclessWwrapper
# >>> env = SpeclessWwrapper(env, states, actions)

Note, continuous space will be supported in the future
(using Sampled-based planners to translate the env into a finite system.)

If wanted, we can extend it to multiple agents
# TODO: >>>
# >>> from specless.minigrid.core import MultiAgentWrapper
# >>> initial_states = [(1, 1), (2, 2), (3, 3)]
# >>> env = MultiAgentWrapper(env, initial_states, concurrent=False) # Turn-based

Transition System Builder
=========================
>>> import numpy as np
>>> from specless.automaton.transition_system import TSBuilder
>>> env = gym.make("MiniGrid-TSP-v0")
>>> env = MiniGridTransitionSystemWrapper(env, ignore_direction=True)
>>> tsbuilder = TSBuilder()
>>> ts = tsbuilder(env)

For multiple agents
>>> env = gym.make("MiniGrid-TSP-v0")
>>> env = MiniGridTransitionSystemWrapper(env, ignore_direction=True)
>>> initial_states = [(1, 1), (2, 2), (3, 3)]

#TODO: >>> env = MultiAgentWrapper(env, initial_states, concurrent=True)

>>> tsbuilder = TSBuilder()
>>> ts = tsbuilder(env)
"""

from typing import Dict, List, Tuple

from gym_minigrid.minigrid import MiniGridEnv
from gymnasium.core import ActType

from specless.const import (
    IDX_TO_COLOR,
    MINIGRID_TO_GRAPHVIZ_COLOR,
    OBJECT_TO_IDX,
)
from specless.wrapper.actionwrapper import (
    FOUR_ACTION_TO_POS_DELTA,
    FourOmniDirectionActions,
    OmniDirectionActionWrapper,
)
from specless.wrapper.labelwrapper import (
    AddPosDirToMiniGridWrapper,
    LabelMiniGridWrapper,
)
from specless.wrapper.tswrapper import TransitionSystemWrapper


class MiniGridTransitionSystemWrapper(TransitionSystemWrapper):
    """Wrapper for the MiniGrid environment to build a transition system.

    MiniGridEnv returns a state of type Dict

    # state: Dict = {
    #      'image': image,
    #      'direction': self.agent_dir,
    #      'mission': self.mission
    # }

    """

    LABELKEY = "observation"

    def __init__(
        self,
        env: MiniGridEnv,
        skip_observations: List[str] = ["unseen", "wall", "empty"],
        ignore_done: bool = True,
        ignore_direction: bool = True,
    ):
        """

        MiniGridEnv
        -----------
        # Action enumeration for this environment
        self.actions: IntEnum = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space: gym.spaces.Discrete = spaces.Discrete(len(self.actions))

        """
        self.ignore_direction = ignore_direction

        if ignore_direction:
            env = OmniDirectionActionWrapper(
                env, FourOmniDirectionActions, FOUR_ACTION_TO_POS_DELTA
            )
        # Label each state using LabelMiniGridWrapper
        env = LabelMiniGridWrapper(
            env,
            labelkey=self.LABELKEY,
            skiplist=skip_observations,
        )
        env = AddPosDirToMiniGridWrapper(env)
        super().__init__(env, ignore_done=ignore_done)

        actions = self.get_wrapped_actions()

        # building some more constant DICTS dynamically from the env data
        ACTION_STR_TO_ENUM = {
            actions._member_names_[action]: action for action in actions
        }
        self.ACTION_ENUM_TO_STR = dict(
            zip(ACTION_STR_TO_ENUM.values(), ACTION_STR_TO_ENUM.keys())
        )

    def _get_action_str(self, action: ActType) -> str:
        return self.ACTION_ENUM_TO_STR[action]

    def _get_node_from_state(self, state: Dict) -> Tuple:
        """Get current "node" from state.
        We only need agent's position and direction.

        Args
        ----
        state (Dict): _description_

        Returns
        -------
        Tuple: _description_
        """
        # Copied from gym_minigrid.minigrid.MiniGridEnv.__str__
        AGENT_DIR_TO_STR: dict[int, str] = {0: ">", 1: "V", 2: "<", 3: "^"}
        if self.ignore_direction:
            return state["pos"]
        return state["pos"], AGENT_DIR_TO_STR[state["dir"]]

    def _get_state(self, obs) -> Dict:
        """Returns the current state.
        AUGMENT ANY NECESSARY INFO that is required in other functions

        MiniGrid Env Observations are dictionaries containing:
            - an image (partially observable view of the environment)
            - the agent's direction/orientation (acting as a compass)
            - a textual mission string (instructions for the agent)

        Additionally, we need the agent's position and the direction

        Returns
        -------
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
        cell = base_env.grid.get(*base_env.agent_pos)

        if cell is None:
            obj_type = OBJECT_TO_IDX["empty"]
            obs = (obj_type, 0, 0)
        else:
            obs = cell.encode()

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

        Args
        ----
        state (Dict): _description_

        Returns
        -------
        _type_: _description_
        """
        return state[self.LABELKEY]
