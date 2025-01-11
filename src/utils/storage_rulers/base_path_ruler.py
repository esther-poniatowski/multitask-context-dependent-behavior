#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`utils.storage_rulers.base_path_ruler` [module]

Manage the file system paths centrally.

Any object which needs to access data on the file system can interact with one PathRuler subclass
(see Implementation).

Classes
-------
`PathRuler` (ABC)

Implementation
--------------
- Arguments
  Path generation methods require minimum string arguments.
- Output Paths
  Paths include only the directories and the file name, without appending any file extension. This
  is handled by the saver and loader methods specific to each file format. Thereby, paths and
  formats are decoupled and can be flexibly combined.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
import os


class PathRuler(ABC):
    """
    Interface to manage file paths.

    Attributes
    ----------
    root_data : str or Path
        Root directory for the data files.
        If not provided, the default is the value of the environment variable `DATA_DIR` defined in
        the system.

    Methods
    -------
    `get_path` (abstract)
    `get_root` (static)
    """

    def __init__(self, root_data: Optional[Union[str, Path]] = None) -> None:
        if root_data is None:
            root_data = self.get_root()
        root_data = Path(root_data)
        if not root_data.is_dir():
            raise FileNotFoundError(f"[ERROR] Invalid root directory: {root_data}")
        self.root_data = root_data

    @abstractmethod
    def get_path(self, *args, **kwargs) -> Path:
        """
        Construct the path to a file. To be implemented in specific subclasses.

        Returns
        -------
        path : Path
            Path to a specific file.
        """

    @staticmethod
    def get_root() -> Path:
        """
        Get the data root directory from the environment variable defined in the system.

        Returns
        -------
        root_data : str
            Root directory for the data files.

        Raises
        ------
        EnvironmentError
            If no environment variable exists under the name `DATA_DIR`.
        """
        root_data = os.environ.get("DATA_DIR")
        if root_data is None:
            raise EnvironmentError("[ERROR] Environment variable 'DATA_DIR' is not set.")
        return Path(root_data)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> ROOT: {self.root_data}"
