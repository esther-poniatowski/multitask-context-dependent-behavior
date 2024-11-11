#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`utils.io_data.loaders_loaders` [module]

Load data from files in specific formats.

Any object which needs to load data can interact with one Loader subclass.

Classes
-------
`LoaderPKL`
`LoaderDILL`
`LoaderNPY`
`LoaderCSVtoList`
`LoaderCSVtoArray`
`LoaderCSVtoArrayFloat`
`LoaderCSVtoArrayInt`
`LoaderCSVtoArrayStr`
`LoaderCSVtoDataFrame`
`LoaderYAML`

See Also
--------
`utils.io_data.base_io.FileExt`
`utils.io_data.base_loader.Loader`

Implementation
--------------
Each Loader subclass implements the abstract method `_load` to load data from a specific file format
to a specific target type.
"""
import pickle
from typing import Any, Union, List, Dict

import csv
import dill
import numpy as np
import pandas as pd
import yaml

from utils.io_data.base_io import FileExt
from utils.io_data.base_loader import Loader


class LoaderPKL(Loader):
    """
    Load any Python object from a Pickle file.

    See Also
    --------
    `pickle.load`
    """

    EXT = FileExt("pkl")

    def _load(self) -> Any:
        """Implement the abstract method of the `Loader` base class."""
        with self.path.open("rb") as file:
            return pickle.load(file)


class LoaderDILL(Loader):
    """
    Load any Python object from a pickle file using the Dill library.

    This module extends the `pickle` module to serialize a larger range of Python objects.

    See Also
    --------
    `dill.load`
    """

    EXT = FileExt("pkl")

    def _load(self) -> Any:
        """Implement the abstract method of the `Loader` base class."""
        with self.path.open("rb") as file:
            return dill.load(file)


class LoaderNPY(Loader):
    """
    Load data from a NPY file to a numpy array.

    See Also
    --------
    `numpy.load`
    """

    EXT = FileExt("npy")

    def _load(self) -> np.ndarray:
        """Implement the abstract method of the `Loader` base class."""
        return np.load(self.path)


class LoaderCSVtoList(Loader):
    """
    Load data from a CSV file to a list of lists.

    See Also
    --------
    `csv.reader`
    """

    EXT = FileExt("csv")

    def _load(self) -> List:
        """Implement the abstract method of the `Loader` base class."""
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return [row for row in reader]


class LoaderCSVtoArray(Loader):
    """
    Load data from a CSV file to a numpy array.

    Class Attributes
    ----------------
    DTYPE : str, default="float"
        Data type of the array contents, passed as argument in the loader function `numpy.loadtxt`.
        Valid values: `int`, `float`, `str`, `complex`, `bool`, `object`.

    Notes
    -----
    This class is refined by three subclasses to specify the data type of the array contents:

    - `LoaderCSVtoArrayFloat`
    - `LoaderCSVtoArrayInt`
    - `LoaderCSVtoArrayStr`

    Those subclasses only redefine the attribute :attr:`DTYPE` to specify the data type.

    Warning
    -------
    If the data type is not specified, the default type is ``float``.

    Unexpected behaviors may occur if the data type is not specified:

    - Integer data is converted to ``float`` (default type).
    - String data raise a ValueError.

    See Also
    --------
    `csv.reader`
    """

    EXT = FileExt("csv")
    DTYPE = "float"

    def _load(self) -> np.ndarray:
        """Load a CSV file into a numpy array.

        Warning
        -------
        The attribute :obj:`tpe` should specify not only ``npt.NDArray``, but also the precise *data
        type* of the array contents.

        - For float data : np.ndarray[np.float64]
        - For integer data : np.ndarray[np.int64]
        - For string data : np.ndarray[np.str_]

        See Also
        --------
        `numpy.loadtxt(file, delimiter, dtype)`
            Load data from a text file. It is used instead of the `csv` module for efficiency: it
            loads the CSV file at once, while the `csv` module reads the file line by line.
        """
        return np.loadtxt(self.path, delimiter=",", dtype=self.DTYPE)


class LoaderCSVtoArrayFloat(LoaderCSVtoArray):
    """Load data from a CSV file to a numpy array of floats."""

    DTYPE = "float"


class LoaderCSVtoArrayInt(LoaderCSVtoArray):
    """Load data from a CSV file to a numpy array of integers."""

    DTYPE = "int"


class LoaderCSVtoArrayStr(LoaderCSVtoArray):
    """Load data from a CSV file to a numpy array of strings."""

    DTYPE = "str"


class LoaderCSVtoDataFrame(Loader):
    """
    Load data from a CSV file to a pandas DataFrame.

    See Also
    --------
    `pandas.read_csv`
    """

    EXT = FileExt("csv")

    def _load(self) -> pd.DataFrame:
        """Implement the abstract method of the `Loader` base class."""
        return pd.read_csv(self.path)


class LoaderYAML(Loader):
    """
    Load data from a YAML file to the corresponding Python object: list or dictionary.

    See Also
    --------
    `yaml.safe_load`: Parse a YAML file and return the corresponding Python object.

    Notes
    -----
    Rules for parsing YAML files:

    - YAML sequences become Python lists
    - YAML mappings become Python dictionaries

    Format of a YAML Block Style Sequence:

    .. code-block:: yaml

        - Item 1
        - Item 2
        - Item 3

    Format of a YAML Block Style Mapping:

    .. code-block:: yaml

        key1: value1
        key2: value2
        key3: value3

    Examples
    --------
    Sequence of Mappings:

    .. code-block:: yaml

            - key1: value1
              key2: value2
            - key1: value3
              key2: value4

    Python object: List of dictionaries

    .. code-block:: python

        [
            {'key1': 'value1', 'key2': 'value2'},
            {'key1': 'value3', 'key2': 'value4'}
        ]

    Mapping of Sequences:

    .. code-block:: yaml

        key1:
        - item1
        - item2
        key2:
        - item3
        - item4

    Python object: Dictionary of lists

    .. code-block:: python

        {
            'key1': ['item1', 'item2'],
            'key2': ['item3', 'item4']
        }
    """

    EXT = FileExt("yml")

    def _load(self) -> Union[Dict, List]:
        """Implement the abstract method of the `Loader` base class."""
        with self.path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
