from typing import Dict, List, Set, Tuple, Union

from gymnasium.core import ActType, ObsType

RewardType = Union[int, float]
MDPStep = Tuple[ObsType, ActType, ObsType, RewardType, bool, Dict]
MDPTrajectory = List[MDPStep]
Symbol = Set[str]
TimeStamp = float
Trace = List[Symbol]
TimedTrace = List[Tuple[Symbol, TimeStamp]]
