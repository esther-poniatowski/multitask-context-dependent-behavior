#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`utils.io_data.base_loader` [module]

Common interface to load data from files.

Classes
-------
`Loader` (ABC, Generic)
"""

from abc import abstractmethod
from pathlib import Path
from typing import Union, Any

from utils.io_data.base_io import IOHandler


class Loader(IOHandler):
    """
    Load data from files in an arbitrary format.

    Class Attributes
    ----------------
    EXT : FileExt
        Extension for the specific file format (see `IOHandler.EXT`).

    Attributes
    ----------
    path : Path
        Path to the file containing the data to load.

    Methods
    -------
    `load`
    `_load` (abstract)

    Raises
    ------
    FileNotFoundError
        If the file to load does not exist.

    Examples
    --------
    Load the content of a CSV file to a DataFrame:

    >>> loader = LoaderCSVtoDataFrame("path/to/data.csv")
    >>> data = loader.load()

    Notes
    -----
    The appropriate file extension is automatically added to the path (base class functionality).

    See Also
    --------
    `utils.io_data.base_io.FileExt`: File extensions.
    `utils.io_data.base_io.IOHandler`: Base class for loading and saving data.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)  # call the constructor of IOHandler
        if not self.server.is_file(path):
            raise FileNotFoundError(f"Inexistent path: {path}")

    def load(self) -> Any:
        """
        Load data from a file.

        Returns
        -------
        data : Any
            Data loaded in the target type.

        Raises
        ------
        Exception
            If the loading process fails.

        See Also
        --------
        `utils.path_system.manage_local.LocalServer.is_file`
            Check the existence of a file at a path in the system.
        """
        try:
            data = self._load()
        except Exception as exc:
            print(f"'{self.__class__.__name__}' failed for path '{self.path}' ")
            raise exc
        return data

    @abstractmethod
    def _load(self) -> Any:
        """Implement the logic to load data from a file in the target format."""
