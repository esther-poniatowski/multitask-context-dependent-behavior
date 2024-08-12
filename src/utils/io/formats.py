#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io.formats` [module]

Classes
-------
:class:`FileExt`
:class:`TargetType`

See Also
--------
:class:`enum.Enum`: Base class for enumeration classes.
"""

from enum import Enum


class FileExt(Enum):
    """Extensions for file formats."""

    CSV = ".csv"
    NC = ".nc"
    NPY = ".npy"
    PKL = ".pkl"


class TargetType(Enum):
    """
    Target types for data loading.

    For usage, each type is specified by the string value, which acts as a descriptive identifier.
    The interest of this enum class is to enforce the use of supported types,
    and to specify a clear correspondence between string identifiers and target types.
    """

    OBJECT = "object"  # any Python object
    LIST = "list"  # flat list or list of lists
    DICT = "dict"  # dictionary
    NDARRAY = "ndarray"  # numpy.ndarray
    NDARRAY_INT = "ndarray_int"  # numpy.ndarray of integers
    NDARRAY_FLOAT = "ndarray_float"  # numpy.ndarray of floats
    NDARRAY_STR = "ndarray_str"  # numpy.ndarray of strings
    DATAFRAME = "dataframe"  # pandas.DataFrame
