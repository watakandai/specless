"""
===============
Strategy module
===============

This module contains the base classes for all strategy classes.

Classes
-------
Strategy
    Base class for all strategy classes.
FeedbackStrategy
    Base class for all feedback strategy classes.
FeedforwardStrategy
    Base class for all feedforward strategy classes.
MemorylessStrategy
    Base class for all memoryless strategy classes.
HistoryDependentStrategy
    Base class for all history-dependent strategy classes.
PlanStrategy
    Plan strategy class.
PolicyStrategy
    Policy strategy class.
CombinedStrategy
    Combined strategy class.

Examples
--------
>>> from gymnasium.strategy import PlanStrategy
>>> plan = [0, 1, 2, 3]
>>> strategy = PlanStrategy(plan)
>>> state = [0, 0, 0]
>>> action = strategy.action(state)
>>> print(action)
[0, 1, 2, 3]
"""

from abc import ABCMeta, abstractmethod
from typing import List

from gymnasium.core import ActType, ObsType


class Strategy(metaclass=ABCMeta):
    """Base class for all strategy classes"""

    def __init__(self) -> None:
        """Initialize the Strategy."""
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset the strategy.

        Raises
        ------
        Exception
            If the method is not implemented in a subclass.
        """
        raise Exception("Must reset")

    @abstractmethod
    def action(self, state: ObsType) -> ActType:
        """Get the action for the strategy.

        Returns
        -------
        Any
            The action for the strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


class FeedbackStrategy(Strategy):
    """Base class for all feedback strategy classes"""

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """
        Reset the feedback strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()

    def action(self, state: ObsType) -> ActType:
        """
        Get the action for the feedback strategy given the state.

        Parameters
        ----------
        state : ObsType
            The state for which to get the action.

        Returns
        -------
        ActType
            The action for the feedback strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


class FeedforwardStrategy(Strategy):
    """Base class for all feedforward strategy classes.

    Attributes
    ----------
    plan : List
        The plan for the feedforward strategy.
    step : int
        The current step in the plan.
    """

    def __init__(self, plan: List) -> None:
        """Initialize the FeedforwardStrategy.

        Parameters
        ----------
        plan : List
            The plan for the feedforward strategy.
        """
        super().__init__()
        self.plan: List = plan
        self.reset()

    def reset(self):
        """Reset the feedforward strategy by setting the current step to 0."""
        self.step: int = 0

    def action(self, state: ObsType) -> ActType:
        """Get the action for the feedforward strategy given the state.

        Parameters
        ----------
        state : ObsType
            The state for which to get the action.

        Returns
        -------
        ActType
            The action for the feedforward strategy, or None if the end of the plan has been reached.
        """
        if len(self.plan) <= self.step:
            return None
        action = self.plan[self.step]
        self.step += 1
        return action


class MemorylessStrategy(FeedbackStrategy):
    """Base class for all memoryless strategy classes."""

    def __init__(self) -> None:
        """Initialize the MemorylessStrategy."""
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        """Get the action for the memoryless strategy given the state.

        Parameters
        ----------
        state : ObsType
            The state for which to get the action.

        Returns
        -------
        ActType
            The action for the memoryless strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


class HistoryDependentStrategy(FeedbackStrategy):
    """Base class for all history-dependent strategy classes."""

    def __init__(self) -> None:
        """Initialize the HistoryDependentStrategy."""
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        """Get the action for the history-dependent strategy given the state.

        Parameters
        ----------
        state : ObsType
            The state for which to get the action.

        Returns
        -------
        ActType
            The action for the history-dependent strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


class PlanStrategy(FeedforwardStrategy):
    """Plan strategy class.
    It ignores all observed states and simply rolls out the pre-computed plan.
    """

    def __init__(self, plan: List) -> None:
        """Initialize the PlanStrategy.

        Parameters
        ----------
        plan : List
            The plan for the plan strategy.
        """
        super().__init__(plan)


class PolicyStrategy(MemorylessStrategy):
    """Policy strategy class.
    It takes an action given an observed state.
    """

    def __init__(self) -> None:
        """Initialize the PolicyStrategy."""
        super().__init__()

    def action(self, state: ObsType) -> ActType:
        """Get the action for the policy strategy given the state.

        Parameters
        ----------
        state : ObsType
            The state for which to get the action.

        Returns
        -------
        ActType
            The action for the policy strategy.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError()


class CombinedStrategy(Strategy):
    """Combined strategy class.
    It takes an action for each strategy given an observed state.
    """

    def __init__(self, strategies: List[Strategy]) -> None:
        """Initialize the CombinedStrategy.

        Parameters
        ----------
        strategies : List[Strategy]
            The list of strategies for the combined strategy.
        """
        self.strategies = strategies
        super().__init__()

    def reset(self) -> None:
        """
        Reset the combined strategy by resetting each strategy in the list.
        """
        map(lambda x: x.reset(), self.strategies)

    def action(self, state: ObsType) -> ActType:
        """Get the action for each strategy in the combined strategy given the state.

        Parameters
        ----------
        state : ObsType
            The state for which to get the action.

        Returns
        -------
        ActType
            The list of actions for the combined strategy.
        """
        return [strategy.action(state) for strategy in self.strategies]
