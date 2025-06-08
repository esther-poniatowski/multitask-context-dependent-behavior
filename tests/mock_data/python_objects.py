"""
:mod:`mock_data.python_objects` [module]

Generate sample data in different formats and structures to test I/O handlers.
"""
from typing import List, Any, Dict, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame


# --- INPUT DATA ---

data_list: List[List[Union[str, int]]] = [
    ["dim1", "dim2"],
    [1, 4],
    [2, 5],
    [3, 6],
]

data_dict: Dict[str, Any] = {
    "dim1": np.array([1, 2, 3]),
    "dim2": np.array([4, 5, 6]),
}

data_array: npt.NDArray = np.array(data_list[1:])  # exclude headers

data_array_float: npt.NDArray[np.float64] = data_array.astype(float)

data_array_int: npt.NDArray[np.float64] = data_array.astype(int)

data_array_str: npt.NDArray[np.float64] = data_array.astype(str)

data_df: DataFrame = pd.DataFrame(data_array, columns=data_list[0])


class MyClass:
    """Sample class to generate objects with complex data."""

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        labels: npt.NDArray[np.str_],
        errors: npt.NDArray[np.bool_],
        meta1: str,
        meta2: List[str],
    ) -> None:
        self.data = data
        self.time = time
        self.labels = labels
        self.errors = errors
        self.metadata1 = meta1
        self.metadata2 = meta2

    def __eq__(self, other):
        if isinstance(other, MyClass):
            for key in self.__dict__:
                if key in ["meta1", "meta2"]:
                    if getattr(self, key) != getattr(other, key):
                        return False
                else:
                    if not np.array_equal(getattr(self, key), getattr(other, key)):
                        return False
        return True


shape = (10, 5, 2)

data_obj = MyClass(
    data=np.random.random(shape),
    time=np.linspace(0, 1, shape[0]),
    labels=np.array([str(i) for i in range(shape[1])]),
    errors=np.array([True, False]),
    meta1="name",
    meta2=["info", "info", "info"],
)


# --- EXPECTED DATA ---

expected_from_list: List[List[str]] = [[str(cell) for cell in row] for row in data_list]
"""Content loaded from a CSV file: all elements are transformed to strings."""
