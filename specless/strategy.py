from abc import ABCMeta, abstractmethod
from typing import List

from gymnasium.core import ActType, ObsType


class Strategy(metaclass=ABCMeta):
    """Base class for all strategy classes"""

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        raise Exception("Must reset")

    @abstractmethod
    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class FeedbackStrategy(Strategy):
    """Base class for all feedback strategy classes"""

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        raise NotImplementedError()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class FeedforwardStrategy(Strategy):
    """Base class for all feedforward strategy classes"""

    def __init__(self, plan: List) -> None:
        super().__init__()
        self.plan: List = plan
        self.reset()

    def reset(self):
        self.step: int = 0

    def action(self, state: ObsType) -> ActType:
        if len(self.plan) <= self.step:
            return None
        action = self.plan[self.step]
        self.step += 1
        return action


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

    def __init__(self, plan: List) -> None:
        super().__init__(plan)


class PolicyStrategy(MemorylessStrategy):
    """Policy strategy class.
    It takes an action given an observed state.
    """

    def __init__(self) -> None:
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        raise NotImplementedError()


class CombinedStrategy(Strategy):
    """Policy strategy class.
    It takes an action given an observed state.
    """

    def __init__(self, strategies: List[Strategy]) -> None:
        self.strategies = strategies
        super().__init__()

    def reset(self) -> None:
        map(lambda x: x.reset(), self.strategies)

    def action(self, state: ObsType) -> ActType:
        return [strategy.action(state) for strategy in self.strategies]
