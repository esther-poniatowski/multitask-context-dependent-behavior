#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io_data.saver_base` [module]

Common interface to save data to files.

The method used for saving is determined automatically from the data type.
The appropriate file extension is automatically added to the path.

Classes
-------
:class:`Saver` (ABC)

Implementation
--------------
The base constructors define the key arguments required by any subclass (path and data).
Subclasses can include additional arguments for their specific operations.
"""

from abc import ABC
from pathlib import Path
from typing import Any, Mapping, Union

from utils.io_data.formats import FileExt
from utils.path_system.local_server import LocalServer


class Saver(ABC):
    """
    Save data to files in an arbitrary format.

    Class Attributes
    ----------------
    ext : FileExt
        File extension for the specific format.
    save_methods : MappingProxyType[type, str]
        Mapping between supported data types and saving methods.
        Keys : Data types
        Values : Method names

    Attributes
    ----------
    data : Any
        Data to save.
    path : Path
        Path to the file where the data will be saved.
    server : LocalServer
        Instance of the local server manager which provides utility file system operations.

    Methods
    -------
    :meth:`save`
    :meth:`_save`
    :meth:`_check_data`

    Raises
    ------
    FileNotFoundError
        If the directory in which to save does not exist.
    TypeError
        If the data type is not supported.
    ValueError
        If the file extension is incorrect.

    See Also
    --------
    :class:`utils.io_data.formats.FileExt`: File extensions.
    :class:`abc.ABC`: Abstract base class.
    :class:`utils.path_system.manage_local.LocalServer`: Utility class.
    """

    ext: FileExt
    save_methods: Mapping[type, str]

    def __init__(self, path: Union[str, Path], data: Any) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.data = data
        self.server = LocalServer()

    def save(self):
        """
        Save data to a file.

        See Also
        --------
        :meth:`utils.path_system.manage_local.LocalServer.check_parent`
            Check the existence of the parent directory.
        :meth:`utils.path_system.manage_local.LocalServer.enforce_ext`
            Enforce the correct file extension.
        """
        self.server.check_parent(self.path)
        self.path = self.server.enforce_ext(self.path, self.ext)
        self._check_data()
        self._save()
        print(f"[SUCCESS] Saved to {self.path}")

    def _save(self):
        """
        Call the appropriate method depending on the type of data.

        Implementation
        --------------
        The method name is identified from the dictionary :attr:`save_methods`.
        - Key: Data type
        - Value : Method name
        The actual method is retrieved by calling :func:`getattr` on the instance.
        """
        method_name = self.save_methods[type(self.data)]
        save_method = getattr(self, method_name)
        save_method()

    def _check_data(self) -> None:
        """
        Check the validity of the data type for the saving format.

        Raises
        ------
        ValueError
            If the data type is not supported, i.e. it is not a key in the :attr:`save_methods`
            dictionary, which means that no method for this format is implemented in the saver.
        """
        if not any(issubclass(type(self.data), tpe) for tpe in self.save_methods):
            raise TypeError(
                f"[ERROR] Unsupported type for {self.ext.value}: {type(self.data).__name__}."
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> - Path: {self.path}"
