from abc import ABCMeta, abstractmethod

from gymnasium.core import ActType, ObsType


class Strategy(metaclass=ABCMeta):
    """Base class for all strategy classes"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class FeedbackStrategy(Strategy):
    """Base class for all feedback strategy classes"""

    def __init__(self) -> None:
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class FeedforwardStrategy(Strategy):
    """Base class for all feedforward strategy classes"""

    def __init__(self) -> None:
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class MemorylessStrategy(FeedbackStrategy):
    """Base class for all memoryless strategy classes"""

    def __init__(self) -> None:
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class HistoryDependentStrategy(FeedbackStrategy):
    """Base class for all history-dependent strategy classes"""

    def __init__(self) -> None:
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class PlanStrategy(FeedforwardStrategy):
    """Plan strategy class.
    It ignores all observed states. Simply rollouts the pre-computed plan.
    """

    def __init__(self) -> None:
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class PolicyStrategy(MemorylessStrategy):
    """Policy strategy class.
    It takes an action given an observed state.
    """

    def __init__(self) -> None:
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()
