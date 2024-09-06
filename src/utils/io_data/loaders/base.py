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
from path_system.local_server import LocalServer


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
    :class:`utils.path_system.manage_local.LocalServer`: Utility class.

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
        self.server = LocalServer()

    def load(self) -> T:
        """
        Load data from a file.

        Returns
        -------
        data: T
            Data loaded in the target type.

        See Also
        --------
        :meth:`utils.path_system.manage_local.LocalServer.enforce_ext`
            Enforce the correct file extension.
        :meth:`utils.path_system.manage_local.LocalServer.is_file`
            Check the existence of a file at a path in the system.
        """
        self.server.is_file(self.path)
        self.path = self.server.enforce_ext(self.path, self.ext)
        self._check_type()
        data = self._load()
        print(f"[SUCCESS] Loaded: {self.path} Type: {self.tpe.value}")
        return data

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
                f"[ERROR] Unsupported target type: {self.tpe} not in {self.load_methods.keys()}"
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> - {self.tpe.value} - Path: {self.path}"
