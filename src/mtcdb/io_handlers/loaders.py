#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.io_handlers.loaders` [module]

Load data from files in specific formats.

Any object which needs to load data can interact with one Loader subclass.

Classes
-------
:class:`LoaderPKL`
:class:`LoaderNPY`
:class:`LoaderCSV`

See Also
--------
:class:`mtcdb.io_handlers.formats.FileExt`
:class:`mtcdb.io_handlers.formats.TargetType`
:class:`mtcdb.io_handlers.loader_base.Loader`
"""

import pickle
from types import MappingProxyType
from typing import Any

import csv
import numpy as np
import numpy.typing as npt
import pandas as pd

from mtcdb.io_handlers.formats import FileExt, TargetType
from mtcdb.io_handlers.loader_base import Loader


class LoaderPKL(Loader):
    """
    Load data from a Pickle file.
    """
    ext = FileExt.PKL
    load_methods = {TargetType.OBJECT: "_load"}

    def __init__(self, path: str) -> None:
        """Override the base class method since the type does not need to be specified."""
        super().__init__(path, tpe='object')

    def _load(self) -> Any:
        """
        Load any Python object from a Pickle file.

        Override the abstract method in the base class
        to load any Python object without requiring a key
        in the dictionary :obj:`load_methods`.
        
        See Also
        --------
        :func:`pickle.load`
        """
        with self.path.open('rb') as file:
            return pickle.load(file)

class LoaderNPY(Loader):
    """
    Load data from a NPY file.
    """
    ext = FileExt.NPY
    load_methods = MappingProxyType({TargetType.NDARRAY: "_load_numpy"})

    def __init__(self, path: str) -> None:
        """Override the base class method since the type is fixed."""
        super().__init__(path, tpe='ndarray')
    
    def _load_numpy(self) -> npt.NDArray:
        """Load a NPY file into a numpy array.
        
        See Also
        --------
        :func:`numpy.load`
        """
        return np.load(self.path)


class LoaderCSV(Loader):
    """
    Load data from a CSV file.
    """
    ext = FileExt.CSV
    load_methods = MappingProxyType({
        TargetType.LIST: "_load_list",
        TargetType.NDARRAY_FLOAT: "_load_numpy",
        TargetType.NDARRAY_INT: "_load_numpy",
        TargetType.NDARRAY_STR: "_load_numpy",
        TargetType.DATAFRAME: "_load_dataframe"
    })

    def _load_list(self) -> list:
        """Load a CSV file into a list of lists.
        
        See Also
        --------
        :func:`csv.reader`
        """
        with self.path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            return [row for row in reader]

    def _load_numpy(self) -> npt.NDArray:
        """Load a CSV file into a numpy array.

        Warning
        -------
        The attribute :obj:`tpe` should specify not only ``npt.NDArray``, 
        but also the precise *data type* of the array contents.
        - For float data : npt.NDArray[np.float64]
        - For integer data : npt.NDArray[np.int64]
        - For string data : npt.NDArray[np.str_]
        If the data type is not specified, the default type is ``float``:
        - Integer data is converted to ``float`` (default type).
        - String data raise a ValueError.
        
        See Also
        --------
        :func:`numpy.loadtxt`
        """
        if self.tpe == TargetType.NDARRAY_FLOAT:
            dtype = 'float'
        elif self.tpe == TargetType.NDARRAY_INT:
            dtype = 'int'
        elif self.tpe == TargetType.NDARRAY_STR:
            dtype = 'str'
        else:
            dtype = 'float' # default type
        return np.loadtxt(self.path, delimiter=",", dtype=dtype)

    def _load_dataframe(self) -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame.
        
        See Also
        --------
        :func:`pandas.read_csv`
        """
        return pd.read_csv(self.path)
