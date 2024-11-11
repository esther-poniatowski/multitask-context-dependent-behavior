#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`utils.io_data.savers` [module]

Save data from files in specific formats.

Any object which needs to save data can interact with one Saver subclass.

Classes
-------
`SaverPKL`
`SaverDILL`
`SaverNPY`
`SaverCSVList`
`SaverCSVArray`
`SaverCSVDataFrame`


See Also
--------
`utils.io_data.base_io.FileExt`: File extensions.
`utils.io_data.base_saver.Saver`: Base class for savers.
"""

import pickle
from typing import Any, List

import csv
import dill
import numpy as np
import pandas as pd

from utils.io_data.base_io import FileExt
from utils.io_data.base_saver import Saver


class SaverPKL(Saver):
    """
    Save any Python object in the Pickle format.

    See Also
    --------
    `pickle.dump`
    """

    EXT = FileExt("pkl")

    def _save(self, data: Any) -> None:
        """Implement the abstract method of the `Loader` base class."""
        with self.path.open("wb") as file:
            pickle.dump(data, file)


class SaverDILL(Saver):
    """
    Save any Python object in the Pickle format using the Dill library.

    Notes
    -----
    This module extends the `pickle` module to serialize a larger range of Python objects.

    Objects handled by the Dill format and not by the Pickle format (examples):

    - Functions and classes.
    - Objects that contain references to functions or classes.
    - Objects that contain circular references.
    - Objects that contain lambda functions, nested functions.
    - Objects that contain NumPy arrays with dtype=object or with numpy.ma module.

    See Also
    --------
    `dill.dump`
    """

    EXT = FileExt("pkl")

    def _save(self, data: Any) -> None:
        """Implement the abstract method of the `Loader` base class."""
        with self.path.open("wb") as file:
            dill.dump(data, file)


class SaverNPY(Saver):
    """
    Save a numpy array in the NPY format.

    See Also
    --------
    `numpy.save`
    """

    EXT = FileExt("npy")

    def _save(self, data: np.ndarray) -> None:
        """Implement the abstract method of the `Loader` base class."""
        np.save(self.path, data)


class SaverCSVList(Saver):
    """
    Save a list of lists to a CSV file.

    See Also
    --------
    `csv.writer`
    """

    EXT = FileExt("csv")

    def _save(self, data: List) -> None:
        """Implement the abstract method of the `Loader` base class."""
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data)


class SaverCSVArray(Saver):
    """
    Save a numpy array to a CSV file.

    Warning
    -------
    The content of the array is written as strings. Thus, when loading the data back, the original data
    type is not preserved.

    Parameter ``fmt``: Format for writing data.

    - ``'%s'``  : strings
    - ``'%d'``  : integers
    - ``'%f'``  : floats

    If not specified, the default format are:

    - ``'%.18e'`` for floating-point numbers
    - string representation for other types

    Thus, differentiation between ``int`` and ``float`` is not automatic.
    Parameter ``delimiter`` : Here, comma (``'``) by default.

    See Also
    --------
    `numpy.savetxt(path, data, delimiter, fmt)`
    """

    EXT = FileExt("csv")

    def _save(self, data: np.ndarray) -> None:
        """Implement the abstract method of the `Loader` base class."""
        if np.issubdtype(data.dtype, np.integer):
            fmt = "%d"
        elif np.issubdtype(data.dtype, np.floating):
            fmt = "%.18e"
        else:
            fmt = "%s"
        np.savetxt(self.path, data, delimiter=",", fmt=fmt)


class SaverCSVDataFrame(Saver):
    """
    Save a pandas DataFrame to a CSV file.

    Class Attributes
    ----------------
    SAVE_INDEX : bool, default=False
        Flag indicating whether to save the DataFrame index as an additional column. It is passed to
        the `to_csv` as the `index` parameter.

    Notes
    -----
    Saving the index is relevant if the index contains meaningful row labels (e.g., timestamps,
    unique identifiers...), but not if the index is just a default integer index. Dropping the index
    ensures that the CSV file format is compatible with other tools that expect data without an
    extra index column. Here, by default, the index is not saved.

    See Also
    --------
    `pandas.DataFrame.to_csv`
    """

    EXT = FileExt("csv")
    SAVE_INDEX = False

    def _save(self, data: pd.DataFrame) -> None:
        data.to_csv(self.path, index=self.SAVE_INDEX)
