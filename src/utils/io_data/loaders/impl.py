#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io_data.loaders.impl` [module]

Load data from files in specific formats.

Any object which needs to load data can interact with one Loader subclass.

Classes
-------
:class:`LoaderPKL`
:class:`LoaderDILL`
:class:`LoaderNPY`
:class:`LoaderCSV`
:class:`LoaderYAML`

See Also
--------
:class:`utils.io_data.formats.FileExt`
:class:`utils.io_data.formats.TargetType`
:class:`utils.io_data.loaders.base.Loader`

Implementation
--------------
When a single target type is supported by a specific loader, it is defined as a class attribute in
the loader class and set as the default value for the :attr:`tpe` parameter in the constructor. It
should match the identifier of the only key in the dictionary :attr:`load_methods`.

This solution is chosen instead of overriding the constructor of the subclass and removing the
:attr:`tpe` parameter from the signature. Indeed, this would not respect the Liskov Substitution
Principle, which is problematic especially in the abstract :class:`Data` class.
"""
import pickle
from types import MappingProxyType
from typing import Any, Union, List, Dict

import csv
import dill
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from utils.io_data.formats import FileExt, TargetType
from utils.io_data.loaders.base import Loader


class LoaderPKL(Loader):
    """Load data from a Pickle file."""

    ext = FileExt.PKL
    tpe = TargetType("object")
    load_methods = {tpe: "_load"}

    def __init__(self, path, tpe=tpe):
        """Override the base class method since the type does not need to be specified."""
        super().__init__(path, tpe=tpe)

    def _load(self) -> Any:
        """
        Load any Python object from a Pickle file.

        Override the abstract method in the base class to load any Python object without requiring a
        key in the dictionary :obj:`load_methods`.

        See Also
        --------
        :func:`pickle.load`
        """
        with self.path.open("rb") as file:
            return pickle.load(file)


class LoaderDILL(Loader):
    """Load data from a Dill file. See :class:`LoaderPKL` for more details."""

    ext = FileExt.PKL
    tpe = TargetType("object")
    load_methods = {tpe: "_load"}

    def __init__(self, path, tpe=tpe):
        """Override the base class method since the type does not need to be specified."""
        super().__init__(path, tpe=tpe)

    def _load(self) -> Any:
        """
        Load any Python object from a Dill file.

        See Also
        --------
        :func:`dill.load`
        """
        with self.path.open("rb") as file:
            return dill.load(file)


class LoaderNPY(Loader):
    """Load data from a NPY file."""

    ext = FileExt.NPY
    tpe = TargetType.NDARRAY
    load_methods = MappingProxyType(
        {
            TargetType.NDARRAY: "_load_numpy",
            TargetType.NDARRAY_FLOAT: "_load_numpy",
            TargetType.NDARRAY_INT: "_load_numpy",
            TargetType.NDARRAY_STR: "_load_numpy",
        }
    )

    def __init__(self, path, tpe=tpe):
        """Override the base class method since the type is fixed."""
        super().__init__(path, tpe=tpe)

    def _load_numpy(self) -> npt.NDArray:
        """Load a NPY file into a numpy array.

        See Also
        --------
        :func:`numpy.load`
        """
        return np.load(self.path)


class LoaderCSV(Loader):
    """Load data from a CSV file."""

    ext = FileExt.CSV
    load_methods = MappingProxyType(
        {
            TargetType.LIST: "_load_list",
            TargetType.NDARRAY_FLOAT: "_load_numpy",
            TargetType.NDARRAY_INT: "_load_numpy",
            TargetType.NDARRAY_STR: "_load_numpy",
            TargetType.DATAFRAME: "_load_dataframe",
        }
    )
    map_data_types = MappingProxyType(
        {
            TargetType.NDARRAY_FLOAT: "float",
            TargetType.NDARRAY_INT: "int",
            TargetType.NDARRAY_STR: "str",
        }
    )

    def _load_list(self) -> List:
        """Load a CSV file into a list of lists.

        See Also
        --------
        :func:`csv.reader`
        """
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            return [row for row in reader]

    def _load_numpy(self) -> npt.NDArray:
        """Load a CSV file into a numpy array.

        Warning
        -------
        The attribute :obj:`tpe` should specify not only ``npt.NDArray``, but also the precise *data
        type* of the array contents.

        - For float data : npt.NDArray[np.float64]
        - For integer data : npt.NDArray[np.int64]
        - For string data : npt.NDArray[np.str_]

        If the data type is not specified, the default type is ``float``:
        - Integer data is converted to ``float`` (default type).
        - String data raise a ValueError.

        See Also
        --------
        :func:`numpy.loadtxt`
        :attr:`map_data_types`
        """
        if self.tpe in self.map_data_types:
            dtype = self.map_data_types[self.tpe]
        else:
            dtype = "float"  # default type
        return np.loadtxt(self.path, delimiter=",", dtype=dtype)

    def _load_dataframe(self) -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame.

        See Also
        --------
        :func:`pandas.read_csv`
        """
        return pd.read_csv(self.path)


class LoaderYAML(Loader):
    """Load data from a YAML file.

    See Also
    --------
    :func:`yaml.safe_load`: Parse a YAML file and return the corresponding Python object.

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

    ext = FileExt.YAML
    load_methods = MappingProxyType(
        {
            TargetType.DICT: "_load",
            TargetType.LIST: "_load",
        }
    )

    def _load(self) -> Union[Dict, List]:
        """Load a YAML file into the corresponding Python object."""
        with self.path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
