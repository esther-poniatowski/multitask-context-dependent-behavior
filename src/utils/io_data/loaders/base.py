#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io_data.loaders.base` [module]

Common interface to load data from files.

Classes
-------
:class:`Loader` (ABC, Generic)
"""

from abc import ABC
from pathlib import Path
from typing import Mapping, Union, TypeVar, Generic

from utils.io_data.formats import FileExt, TargetType
from utils.path_system.explorer import check_path, is_file, enforce_ext


T = TypeVar("T")
"""Type variable representing the arbitrary type of data handled by the Loader."""


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
    path: Path
        Path to the file containing the data to load.
    tpe: TargetType
        Target type for the retrieved data.
        It determines the method used to load the data and the type of the returned object.

    Methods
    -------
    :meth:`load`
    :meth:`_load`
    :meth:`_check_type`

    Raises
    ------
    TypeError
        If the target type is not valid (:attr:`tpe`).
    FileNotFoundError
        If the file to load does not exist.

    See Also
    --------
    :class:`utils.io_data.formats.FileExt`: File extensions.
    :class:`utils.io_data.formats.TargetType`: Target types.
    :class:`abc.ABC`: Abstract base class.
    :func:`utils.path_system.explorer.check_path`: Check the existence of a path in the file system.
    :func:`utils.path_system.explorer.enforce_ext`: Enforce the correct file extension.

    Notes
    -----
    In the constructor parameters, target types are specified by strings identifiers rather than
    actual TargetType objects, for facilitating the instantiation of the loader. This identifier is
    used to instantiate the TargetType object in the constructor.
    To select the appropriate identifier for a target type, inspect the attributes of the class
    :class:`TargetType` and choose one of the types used by the specific loader subclass.
    """

    ext: FileExt
    load_methods: Mapping[TargetType, str]

    def __init__(self, path: Union[str, Path], tpe: Union[str, TargetType]):
        if isinstance(path, str):
            path = Path(path)
        if isinstance(tpe, str):
            tpe = TargetType(tpe)
        self.path = path
        self.tpe = tpe

    def load(self) -> T:
        """
        Load data from a file.

        Returns
        -------
        T
            Data loaded in the target type.
        """
        check_path(self.path)
        is_file(self.path)
        self.path = enforce_ext(self.path, self.ext)
        print(f"Load: {self.path}")
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

    def _check_type(self):
        """
        Check if the target type is supported.

        Raises
        ------
        ValueError
            If the target type is not supported, it is not a key in the dictionary
            :obj:`load_methods`, which means that the method to load this format is not implemented
            in the loader.
        """
        if self.tpe not in self.load_methods:
            raise TypeError(
                f"Unsupported target type: {self.tpe} not in {self.load_methods.keys()}"
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> - {self.tpe.value} - Path: {self.path}"
