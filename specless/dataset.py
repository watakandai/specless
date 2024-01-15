"""
Data object (It's a pandas.DataFrame!!! Just renamed it to Data object)
==============

It's basically a table. You can access its  size:
>> l = demonstration.size       # Return the number of elements in this object


If it were a TimedTraceData object, it has a trace and timestamp data.
>> symbols = demonstration["symbol"]            # or demonstration.symbol
                                                # Returns a Series object
>> timestamps = demonstration"timestamp"]       # or demonstration.timestamp
                                                # Returns a Series object

or turn it into a list of tuples
>> demo_list = demonstration.values.tolist()    # Returns a list of list
>> [[s1, t1], [s2, t2], ..., [sn, tn]]

You can sort the data
>> sorted_demonstration = demonstration.sort_values(by="timestamp")
>> demonstration.sort_values(by=["timestamp", "symbol"], inplace=True)

Dataset object (Followed the PyTorch's Dataset class)
==============
A Data object can access a data (demonstration/trace) by:
>> demonstrations = [
    ["e1", "e2", "e3", "e4", "e5"],             # trace 1
    ["e1", "e4", "e2", "e3", "e5"],             # trace 2
    ["e1", "e2", "e4", "e3", "e5"],             # trace 3
]
>> demonstrations = sl.ArrayDataset(demnstrations, columns=["symbol"])
>> demonstration = demonstrations[i]

We can also return a list of data
>> timed_traces = demonstrations.tolist()
>> [[[s1, t1], [s2, t2], ..., [sn, tn]], ..., [[s1, t1], [s2, t2], ..., [sm, tm]]]
>> traces = demonstrations.tolist(key="symbol")
>> [[s1, s2, ..., sn], ..., [s1, s2, ..., sn]]

You can sort dataset in a batch
>> f = lambda data: data.sort_values(by=["timestamp", "symbol"], inplace=True)
>> demonstrations.apply(f)

"""
import os
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from specless.typing import Data


class BaseDataset(Dataset):
    def __init__(self, data: List[Data]):
        # Check if they have same columns (There must be an unique column)
        assert len(set(map(lambda d: tuple(d.columns.tolist()), data))) == 1
        self.data: List[Data] = data

    @property
    def length(self):
        return len(self.data)

    @abstractmethod
    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

    def apply(self, func: Callable[..., Any]) -> None:
        list(map(func, self.data))

    def tolist(self, key: Optional[str] = None):
        if key is None:
            return list(map(lambda d: d.values.tolist(), self.data))
        else:
            return list(map(lambda d: d[key].values.tolist(), self.data))


class ArrayDataset(BaseDataset):
    def __init__(
        self, data: Union[List[List[Any]], np.ndarray], columns: List[str]
    ) -> None:
        super().__init__([Data(d, columns=columns) for d in data])


class CSVDataset(BaseDataset):
    """Reads a list of csv files and turns them into a dataset"""

    def __init__(
        self,
        filedir: Optional[str] = None,
        filepaths: Optional[List[str]] = None,
    ) -> None:
        """
        Users can either provide a directory to the traces
        or a list of filepaths
        >> filedir = /path/to/the/csv/directory/
        >> dataset = CSVDataset(filedir)

        If the user strictly needs a sorted dataset, they can provide a list of paths
        >> filedir = /path/to/the/csv/directory/
        >> filepaths = [os.path.join(filedir, i) for i in range(100)]
        >> dataset = CSVDataset(filedir, filepaths=filepaths)

        Args:
            filedir (str, optional): _description_. Defaults to None.
        """
        if filedir is None and filepaths is None:
            raise Exception("Provide either filedir or filepaths")

        # List a set of file paths
        def iscsv(f) -> bool:
            return f.endswith(".csv")

        if filepaths is None:
            filepaths = [
                os.path.join(filedir, f)
                for f in os.listdir(filedir)
                if os.path.isfile(os.path.join(filedir, f)) and iscsv(f)
            ]

        super().__init__([pd.read_csv(fp) for fp in filepaths])


class PathToFileDataset(BaseDataset):
    """This class doesn't contain any data,
    but a path to a file tha tcontains the data,
    in case some algorithms require only the filename.
    """

    def __init__(self, filepath: str) -> None:
        super().__init__(data=[])
        self.filepath: str = filepath
