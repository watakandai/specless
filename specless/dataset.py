"""
Data object
==============
Assume demonstrations are defined as a list of Data objects:

Data class functions as a dictionary and a struct, so an item of interest can be accessed via:
>> l = demonstration["length"]
or
>> l = demonstration.length

If it were a TimedTraceData object, it has a trace and timestamp data.
>> symbols = demonstration["symbol"]            # or demonstration.symbol
>> timestamps = demonstration["timestamp"]      # or demonstration.timestamp

or turn it into a list of tuples
>> demo_list = demonstration.to_list()          # sorted by the alphabetical order
>> [(s1, t1), (s2, t2), ..., (sn, tn)]

You can also sort the data
>> sorted_demonstration = demonstration.sort_by("timestamp", inplace=False)
>> demonstration.sort_by("timestamp", inplace=True)

You can also call it in batches
>> f = lambda data: data.sort_by("timestamp", inplace=True)
>> demonstrations.apply(f)

Dataset object
==============
A Data object can access a data by:
>> demonstrations = TraceDataset()
>> demonstration = demonstrations[i]

We can also return a list of data
>> timed_traces = demonstrations.to_list()
>> [[(s1, t1), (s2, t2), ..., (sn, tn)], ..., [(s1, t1), (s2, t2), ..., (sm, tm)]]
>> traces = demonstrations.to_list(key="symbol")
>> [[s1, s2, ..., sn], ..., [s1, s2, ..., sn]]

"""
from abc import ABCMeta, abstractmethod
from typing import Any, List

from specless.typing import MDPTrajectory, Trace


class Dataset(metaclass=ABCMeta):
    """A baseclass for dataset.
    It keeps a list of data.
    The inheriting classes must implement  __len__ and __getitem__ functions.
    """

    # TODO: Replace Any with class T
    def __init__(self, data: List[List[Any]]) -> None:
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
        super().__init__(data=[])
        self.filepath: str = filepath
