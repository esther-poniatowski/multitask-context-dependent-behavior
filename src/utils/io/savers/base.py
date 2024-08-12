#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io.saver_base` [module]

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

from utils.io.formats import FileExt


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
    path_file : Path
        Path to the file.

    Methods
    -------
    :meth:`save`
    :meth:`_save`
    :meth:`_check_dir`
    :meth:`_check_ext`
    :meth:`_check_data`

    See Also
    --------
    :class:`utils.io.formats.FileExt`: File extensions.
    :class:`abc.ABC`: Abstract base class.
    """

    ext: FileExt
    save_methods: Mapping[type, str]

    def __init__(self, path: Union[str, Path], data: Any) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.data = data

    def save(self):
        """Save data to a file."""
        self._check_dir()
        self._check_ext()
        self._check_data()
        self._save()
        print(f"Saved to {self.path}")

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

    def _check_dir(self):
        """
        Check the existence of the saving directory.

        Raises
        ------
        FileNotFoundError
            If the directory in which to save does not exist.

        See Also
        --------
        :attr:`pathlib.Path.parent`: Get the parent directory of one file.
        :meth:`pathlib.Path.exists`: Check if a directory exists in the file system.
        """
        if not self.path.parent.exists():
            raise FileNotFoundError(f"Inexistent directory: {self.path.parent}.")

    def _check_ext(self) -> None:
        """
        Check the validity of the file extension for the specific format.

        If the file extension is missing or incorrect, it is added or corrected.

        Implementation
        --------------
        The extension is retrieved from the class attribute :attr:`ext`,
        by calling the :attr:`value` on the enum object.

        See Also
        --------
        :attr:`pathlib.Path.suffix`
            Get the file extension.
            If there is not extension, the empty string is returned.
        :meth:`pathlib.Path.with_suffix`
            Replace the file extension.
            If there is already an extension, it is replaced.
            If there is no extension, it is added.
        """
        if self.path.suffix != self.ext.value:
            self.path = self.path.with_suffix(self.ext.value)

    def _check_data(self) -> None:
        """
        Check the validity of the data type for the saving format.

        Raises
        ------
        ValueError
            If the data type is not supported, it is not a key
            in the :attr:`save_methods` dictionary, which means that the
            method to save this format is not implemented in the saver.
        """
        if not any(issubclass(type(self.data), tpe) for tpe in self.save_methods):
            raise TypeError(f"Unsupported type for {self.ext.value}: {type(self.data).__name__}.")
