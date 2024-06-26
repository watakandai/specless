"""

Type Definitions
================

This module contains type definitions for the various data structures used in the
specless package.

"""

from enum import IntEnum
from typing import Dict, Iterable, List, Set, Tuple, Union

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from pandas import DataFrame

EnvType = gym.Env
EnvObs = Dict
CellObs = Tuple[int, int, int]
ActionsEnum = IntEnum
Reward = float
Done = bool
StepData = Tuple[EnvObs, Reward, Done, Done, dict]

EnvAct = ActionsEnum
EnvActs = Iterable[ActionsEnum]

RewardType = Union[int, float]
MDPStep = Tuple[ObsType, ActType, ObsType, RewardType, bool, Dict]
MDPTrajectory = List[MDPStep]
Symbol = Set[str]
TimeStamp = float
Trace = List[Symbol]
TimedTrace = List[Tuple[Symbol, TimeStamp]]
TimedTraceList = List[TimedTrace]
TimeBound = Tuple[float, float]
NodeName = str
EdgeName = Tuple[str, str]
NodeBoundDict = Dict[NodeName, TimeBound]
EdgeBoundDict = Dict[EdgeName, TimeBound]
LowerBoundInt = int
UpperBoundInt = int
Data = DataFrame
