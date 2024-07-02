#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.io_handlers.loader_base` [module]

Common interface to load data from files.

Classes
-------
:class:`Loader` (ABC, Generic)
"""

from abc import ABC
from pathlib import Path
from typing import Mapping, TypeVar, Generic

from mtcdb.io_handlers.formats import FileExt, TargetType


T = TypeVar('T')
"""
Type variable representing the arbitrary type of data handled by the Loader.
"""


class Loader(ABC, Generic[T]):
    """
    Load data from files in an arbitrary format.

    Class Attributes
    ----------------
    ext : FileExt
        File extension for the specific format.
    load_methods : Dict[str, str]
        Mapping between target data types and loading methods.
        Keys : String identifiers for the target type.
        Values : Method names.

    Attributes
    ----------
    path : Path
        Path to the file containing the data to load.
    tpe : TargetType
        Target type for the retrieved data.
        It determines the method used to load the data and the type 
        of the returned object.

    Methods
    -------
    :meth:`load`
    :meth:`_load`
    :meth:`_check_file`
    :meth:`_check_type`

    Raises
    ------
    TypeError
        If the target type is not valid.

    See Also
    --------
    :class:`mtcdb.io_handlers.formats.FileExt`: File extensions.
    :class:`mtcdb.io_handlers.formats.TargetType`: Target types.
    :class:`abc.ABC`: Abstract base class.

    Notes
    -----
    In the constructor parameters, target types are specified by strings,
    which are more straightforward to manipulate than actual TargetType objects.
    The string should be one of the values defined in the TargetType enum class.
    This string is used to instantiate the TargetType object in the constructor.
    If the string is not a valid target type, a TypeError is raised.
    To select the appropriate identifier for a target type,
    inspect the attributes of the class :class:`TargetType` 
    and choose one of the types used by the specific loader subclass.
    """
    ext: FileExt
    load_methods: Mapping[TargetType, str]

    def __init__(self, path: str, tpe: str) -> None:
        self.path = Path(path)
        self.tpe = TargetType(tpe)

    def load(self) -> T:
        """
        Load data from a file.
        
        Returns
        -------
        T
            Data loaded in the target type.
        """
        self._check_file()
        self._check_type()
        return self._load()

    def _load(self) -> T:
        """
        Call the appropriate method to load the data based on the specified type.
        
        Returns
        -------
        T
            Data loaded in the target type using the appropriate method.
        """
        method_name = self.load_methods[self.tpe]
        load_method = getattr(self, method_name)
        return load_method()

    def _check_file(self):
        """
        Check if the specified file exists.

        Raises
        ------
        FileNotFoundError
            If the file is not found.

        See Also
        --------
        :func:`pathlib.Path.exists`: Check if a file exists in the file system.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Inexistent file: {self.path}")

    def _check_type(self):
        """
        Check if the target type is supported.

        Raises
        ------
        ValueError
            If the target type is not supported, it is not a key
            in the dictionary :obj:`load_methods`, which means that the 
            method to load this format is not implemented in the loader.
        """
        if self.tpe not in self.load_methods:
            raise TypeError(f"Unsupported target type: {self.tpe}.")
