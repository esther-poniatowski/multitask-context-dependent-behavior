#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io.path_manager_base` [module]

Manage the file system paths centrally.

Any object which needs to access data on the file system can
interact with one PathManager subclass (see Implementation).

Classes
-------
:class:`PathManager` (ABC)

Implementation
--------------
- Arguments
  Path generation methods require minimum string arguments.
- Output Paths
  Paths include only the directories and the file name, without appending any file extension.
  This is handled by the saver and loader methods specific to each file format.
  Thereby, paths and formats are decoupled and can be combined.

See Also
--------
:class:`abc.ABC`: Abstract base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
import os


class PathManager(ABC):
    """
    Interface to manage file paths.

    Attributes
    ----------
    root_data: str or Path
        Root directory for the data files.
        If not provided, the default is the value of the environment variable `DATA_DIR` defined in
        the system.

    Methods
    -------
    :meth:`get_path` (abstract)
    :meth:`get_root` (static)
    :meth:`check_dir` (static)
    :meth:`create_dir` (static)
    :meth:`display_tree`
    """

    def __init__(self, root_data: Optional[Union[str, Path]] = None) -> None:
        if root_data is None:
            root_data = self.get_root()
        root_data = Path(root_data)
        self.check_dir(root_data, raise_error=True)
        self.root_data = root_data

    @abstractmethod
    def get_path(self, *args, **kwargs) -> Path:
        """
        Construct the path for a file. To be implemented in specific subclasses.

        Returns
        -------
        path: str
            Path to a specific file.
        """

    @staticmethod
    def get_root() -> Path:
        """
        Get the data root directory from the environment variable defined in the system.

        Returns
        -------
        root_data: str
            Root directory for the data files.

        Raises
        ------
        EnvironmentError
            If no environment variable exists under the name `DATA_DIR`.
        """
        root_data = os.environ.get("DATA_DIR")
        if root_data is None:
            raise EnvironmentError("Environment variable 'DATA_DIR' is not set.")
        return Path(root_data)

    @staticmethod
    def check_dir(path: Union[str, Path], raise_error: bool = False) -> bool:
        """
        Ensure the existence of a directory in the file system.

        Parameters
        ----------
        path: str or Path
            Path to a directory.
        raise_error: bool, default=False
            Whether to raise an error if the directory does not exist.

        Returns
        -------
        bool
            True if the directory exists, False otherwise.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist and `raise_error` is True.

        See Also
        --------
        :func:`pathlib.exists`: Check if a path exists in the file system.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            if raise_error:
                raise FileNotFoundError(f"Inexistent directory: {path}")
            return False
        return True

    @staticmethod
    def create_dir(path: Union[str, Path]) -> Path:
        """
        Create a directory if it does not exist.

        Parameters
        ----------
        path: str or Path
            Path to a directory to create.

        See Also
        --------
        :func:`pathlib.mkdir`: Create a directory.
            Parameter `parents` : Create parent directories if needed.
            Parameter `exist_ok` : If the directory already exists, nothing is done.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created: {path}")
        else:
            print(f"Pre-existing directory: {path}")
        return path

    def display_tree(
        self, path: Optional[Union[str, Path]] = None, level: int = 0, limit: int = 5
    ) -> None:
        """
        Display the tree structure of a directory.

        Parameters
        ----------
        path : str or Path, optional
            Path to the directory to display. If None, use the root directory.
        level : int, optional
            Current level in the directory tree, used for indentation.
        limit : int, optional
            Maximum number of items to display per directory.

        Implementation
        --------------

        - Display the name of each file or subdirectory in the currently traversed directory, with
          an indentation level depending on its depth in the hierarchy.
        - If a directory contains more items than the specified limit, show an ellipsis (`...`).
        - Call the method recursively on sub-directories until reaching the end of the directory
          hierarchy (i.e when encountering a directory that contains no subdirectories or files).
        """
        if path is None:
            path = self.root_data
        path = Path(path)
        self.check_dir(path, raise_error=True)
        items = list(path.iterdir())
        display_items = items[:limit]
        for item in display_items:
            print("    " * level + "|-- " + item.name)
            if item.is_dir():
                self.display_tree(item, level + 1, limit)
        if len(items) > limit:
            print("    " * level + "|-- ...")