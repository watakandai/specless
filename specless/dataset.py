from abc import ABCMeta, abstractmethod
from typing import Any, List

from specless.typing import MDPTrajectory, Trace


class Dataset(metaclass=ABCMeta):
    """A baseclass for dataset.
    It keeps a list of data.
    The inheriting classes must implement  __len__ and __getitem__ functions.
    """

    def __init__(self, data: List[List[Any]] | None) -> None:
        self.data: List[List[Any]] = data

    @abstractmethod
    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]


class MDPDataset(Dataset):
    """Dataset that stores MDP data.
    It keeps a list of MDP trajectory where a trajectory is defined
    by a sequence of tuples (state, action, next_state, reward, done, info)
    """

    def __init__(self, data: List[MDPTrajectory]) -> None:
        super().__init__(data)


class TraceDataset(Dataset):
    """Dataset that stores trace data.
    It keeps a list of traces where a trace is defined
    by a sequence of symbol, a set of symbol.
    For example a symbol ${a, b, c}$
    """

    def __init__(self, data: List[Trace]) -> None:
        super().__init__(data)


class TimedTraceDataset(Dataset):
    """Dataset that stores timedtrace data.
    It keeps a list of timedtraces where a timedtrace is defined
    by a sequence of tuple (symbol, timestamp).
    """

    def __init__(self, data: List[Trace]) -> None:
        super().__init__(data)


class PathToFileDataset(Dataset):
    """This class doesn't contain any data,
    but a path to a file tha tcontains the data,
    in case some algorithms require only the filename.
    """

    def __init__(self, filepath: str) -> None:
        super().__init__(data=None)
        self.filepath: str = filepath
