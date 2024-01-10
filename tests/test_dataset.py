from typing import List

import numpy as np

from specless.dataset import ArrayDataset, BaseDataset, CSVDataset, Data


def test_data():
    # Instantiate the data (table) by providing a dictionary
    demonstration = Data({"col1": [1, 2], "col2": [3, 4]})
    assert True
    # Instantiate the data (table) by providing a dictionary
    demonstration = Data(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )
    assert True


def test_basedataset():
    # col1 col2
    # 1     3
    # 2     4
    demonstration1 = Data({"col1": [1, 2], "col2": [3, 4]})
    # col1 col2
    # 1     2
    # 3     4
    # 5     6
    # 7     8
    demonstration2 = Data(
        np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), columns=["col1", "col2"]
    )
    dataset = BaseDataset([demonstration1, demonstration2])
    assert True

    assert dataset.length == 2
    assert len(dataset) == 2
    assert dataset[0].equals(demonstration1)
    assert dataset[1].equals(demonstration2)
    assert dataset.tolist() == [[[1, 3], [2, 4]], [[1, 2], [3, 4], [5, 6], [7, 8]]]
    assert dataset.tolist("col1") == [
        [1, 2],
        [1, 3, 5, 7],
    ]
    assert dataset.tolist("col2") == [
        [3, 4],
        [2, 4, 6, 8],
    ]
    # assert dataset.apply()
    f = lambda data: data.sort_values(by="col1", inplace=True, ascending=False)
    dataset.apply(f)
    assert dataset.tolist() == [[[2, 4], [1, 3]], [[7, 8], [5, 6], [3, 4], [1, 2]]]


def test_arraydataset():
    # Similarly, Instantiate a dataset
    demonstrations: list = [
        [[1, "a"], [2, "b"], [3, "c"]],
        [[4, "d"], [5, "e"], [6, "f"]],
    ]
    columns: list = ["timestamp", "symbol"]
    dataset = ArrayDataset(demonstrations, columns)
    assert True

    dataset = ArrayDataset(np.array(demonstrations), columns)
    assert True


def test_csvdataset():
    # First create a temporary csv file
    demonstration = Data(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )
    demonstration.to_csv("/tmp/test1.csv")
    demonstration.to_csv("/tmp/test2.csv")

    # Similarly, Instantiate a dataset
    filedir = "/tmp"
    dataset = CSVDataset(filedir)
    assert True

    filepaths: List[str] = ["/tmp/test1.csv", "/tmp/test2.csv"]
    dataset = CSVDataset(filedir, filepaths=filepaths)
    assert True
