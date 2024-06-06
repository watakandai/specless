"""
========================
Data and Dataset classes
========================

Data Class
==========

It's basically a table. You can access its  size:

>>> from specless.typing import Data
>>> demonstration = Data([['a', 1], ['b', 4], ['c', 6]], columns=['symbol', 'timestamp'])
>>> l = demonstration.size       # Return the number of elements in this object

If it were a TimedTraceData object, it has a trace and timestamp data.

>>> symbols = demonstration["symbol"]            # or demonstration.symbol
...                                              # Returns a Series object

>>> timestamps = demonstration["timestamp"]      # or demonstration.timestamp
...                                              # Returns a Series object

or turn it into a list of tuples

>>> demonstration.values.tolist()                # Returns a list of list
[['a', 1], ['b', 4], ['c', 6]]

You can sort the data

>>> sorted_demonstration = demonstration.sort_values(by="timestamp")
>>> demonstration.sort_values(by=["timestamp", "symbol"], inplace=True)

Dataset Class
==============
A Data object can access a data (demonstration/trace) by:

>>> import specless as sl
>>> demonstrations = [
...     [["e1",1], ["e2",2], ["e3",3], ["e4",4], ["e5",5]],  # trace 1
...     [["e1",1], ["e4",3], ["e2",5], ["e3",7], ["e5",9]],  # trace 2
...     [["e1",2], ["e2",4], ["e4",6], ["e3",8], ["e5",10]], # trace 3
... ]

>>> demonstrations = sl.ArrayDataset(demonstrations, columns=["symbol", "timestamp"])
>>> demonstration = demonstrations[0]

We can also return a list of data

>>> demonstrations.tolist()
[[['e1', 1], ['e2', 2], ['e3', 3], ['e4', 4], ['e5', 5]], [['e1', 1], ['e4', 3], ['e2', 5], ['e3', 7], ['e5', 9]], [['e1', 2], ['e2', 4], ['e4', 6], ['e3', 8], ['e5', 10]]]

>>> demonstrations.tolist(key="symbol")
[['e1', 'e2', 'e3', 'e4', 'e5'], ['e1', 'e4', 'e2', 'e3', 'e5'], ['e1', 'e2', 'e4', 'e3', 'e5']]

You can sort dataset in a batch

>>> f = lambda data: data.sort_values(by=["timestamp", "symbol"], inplace=True)
>>> demonstrations.apply(f)
"""

import glob
import os
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from specless.typing import Data


class BaseDataset(Dataset):
    """Base Dataset Class

    Attributes
    ----------
    """

    def __init__(self, data: List[Data]):
        """Initialize the BaseDataset with a list of data.

        Parameters
        ----------
        data : List[Data]
            A list of data to be stored in the dataset.
        """
        # assert len(set(map(lambda d: tuple(d.columns.tolist()), data))) == 1
        self.data: List[Data] = data

    @property
    def length(self) -> int:
        """Get the length of the data.

        Returns
        -------
        int
            The length of the data.
        """
        return len(self.data)

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the data.

        Returns
        -------
        int
            The length of the data.
        """
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Get the item at the specified index.

        Parameters
        ----------
        idx : int
            The index of the item to get.

        Returns
        -------
        Any
            The item at the specified index.
        """
        return self.data[idx]

    def apply(self, func: Callable[..., Any]) -> None:
        """Apply a function to each item in the data.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to apply to each item.
        """
        list(map(func, self.data))

    def tolist(self, key: Optional[str] = None):
        """Convert the data to a list.

        Parameters
        ----------
        key : Optional[str], default=None
            The key to use to convert the data to a list. If None, all data is converted.

        Returns
        -------
        list
            The data converted to a list.
        """
        if key is None:
            return list(map(lambda d: d.values.tolist(), self.data))
        else:
            return list(map(lambda d: d[key].values.tolist(), self.data))


class ArrayDataset(BaseDataset):
    """Dataset class that contains a list of data."""

    def __init__(
        self, data: Union[List[List[Any]], np.ndarray], columns: List[str]
    ) -> None:
        """Initialize the ArrayDataset with a list of data and column names.

        The data is converted into a list of Data objects, each with the specified columns.

        Parameters
        ----------
        data : Union[List[List[Any]], np.ndarray]
            The data to be stored in the dataset. Can be a list of lists or a numpy array.
        columns : List[str]
            The names of the columns in the data.

        Examples
        --------
        >>> data = [[1, 2, 3], [4, 5, 6]]
        >>> columns = ['symbol']
        >>> dataset = ArrayDataset(data, columns)
        """
        super().__init__([Data(d, columns=columns) for d in data])


class CSVDataset(BaseDataset):
    """Reads a list of csv files and turns them into a dataset.

    Users can either provide a directory to the traces or a list of filepaths.

    Examples
    --------
    Provide a directory to the traces:

    >>> filedir = '/path/to/the/csv/directory/'     # doctest: +SKIP
    >>> dataset = CSVDataset(filedir)               # doctest: +SKIP

    If the user strictly needs a sorted dataset, they can provide a list of paths:

    >>> filedir = '/path/to/the/csv/directory/'                         # doctest: +SKIP
    >>> filepaths = [os.path.join(filedir, str(i)) for i in range(100)]  # doctest: +SKIP
    >>> dataset = CSVDataset(filedir, filepaths=filepaths)                # doctest: +SKIP
    """

    def __init__(
        self,
        filedir: str,
        filepaths: Optional[List[str]] = None,
    ) -> None:
        """Initialize the CSVDataset with a directory or a list of filepaths.

        Parameters
        ----------
        filedir : Optional[str], default=None
            The directory containing the csv files to be included in the dataset.
        filepaths : Optional[List[str]], default=None
            A list of filepaths to the csv files to be included in the dataset.
        """
        if filepaths is None:
            filepaths = sorted(glob.glob(os.path.join(filedir, "*.csv")))
        super().__init__([pd.read_csv(fp) for fp in filepaths])


class PathToFileDataset(BaseDataset):
    """Dataset class that contains a path to a file

    A dataset that contains a path to a file, but not the data itself.

    This can be useful for algorithms that require only the filename, not the data.

    Examples
    --------
    >>> filepath = 'examples/demo/pdfa.yaml'
    >>> dataset = PathToFileDataset(filepath)
    """

    def __init__(self, filepath: str) -> None:
        """Initialize the PathToFileDataset with a filepath.

        Parameters
        ----------
        filepath : str
            The path to the file.
        """
        super().__init__(data=[])
        self.filepath: str = filepath
