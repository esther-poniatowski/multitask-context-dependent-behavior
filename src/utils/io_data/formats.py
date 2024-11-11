#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io_data.formats` [module]

Classes
-------
:class:`TargetType`

See Also
--------
:class:`enum.Enum`: Base class for enumeration classes.
"""

from enum import Enum


class TargetType(Enum):
    """
    Target types for data loading.

    Goals:

    - Enforce the use of supported types for data loading.
    - Specify a clear correspondence between string identifiers and target types.

    Usage:
    Each type is specified by the string value, which acts as a descriptive identifier.
    """

    OBJECT = "object"  # any Python object
    LIST = "list"  # flat list or list of lists
    DICT = "dict"  # dictionary
    NDARRAY = "ndarray"  # numpy.ndarray
    NDARRAY_INT = "ndarray_int"  # numpy.ndarray of integers
    NDARRAY_FLOAT = "ndarray_float"  # numpy.ndarray of floats
    NDARRAY_STR = "ndarray_str"  # numpy.ndarray of strings
    DATAFRAME = "dataframe"  # pandas.DataFrame
