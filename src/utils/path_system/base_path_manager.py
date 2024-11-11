#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.path_system.base_path_manager` [module]

Classes
-------

"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional


class ServerInterface(ABC):
    """
    Abstract base class for managing interactions with a server (local or remote).

    Class Attributes
    ----------------
    default_root : Path
        Default root directory path on the remote server: name of the project in the user's home.

    Methods
    -------
    :meth:`build_path`
    :meth:`is_dir` (abstract)
    :meth:`is_file` (abstract)
    :meth:`check_parent` (abstract)
    :meth:`create_dir` (abstract)
    """

    default_root = Path("~/mtcdb")

    def __init__(self, root_path: Optional[Union[Path, str]] = None):
        if root_path is not None:
            self.root_path = Path(root_path)
        else:
            self.root_path = self.default_root

    def build_path(self, path) -> Path:
        """
        Build a full path on the server from the root path.

        Arguments
        ---------
        path : Path
            Relative path on the remote server, from the root directory of the workspace.

        Returns
        -------
        full_path: Path
            Full path on the remote server, encompassing the root path.
        """
        return self.root_path / path

    @abstractmethod
    def is_dir(self, path: Union[Path, str]) -> bool:
        """
        Check whether a path corresponds to a directory in the file system.

        Parameters
        ----------
        path: str or Path
            Full path to the directory.

        Returns
        -------
        bool
            True if the path exists and is a directory, False otherwise.
        """

    @abstractmethod
    def is_file(self, path: Union[Path, str]) -> bool:
        """
        Check whether a path corresponds to a file in the file system.

        Parameters
        ----------
        path: str or Path
            Full path to the file.

        Returns
        -------
        bool
            True if the path exists and is a file, False otherwise.
        """

    def check_parent(self, path: Union[str, Path]) -> bool:
        """
        Check the existence of the parent directory of a file or other directory.

        Parameters
        ----------
        path: str or Path
            Full path to the file or directory whose parent is to be checked.

        Returns
        -------
        bool
            True if the parent directory exists, False otherwise.

        See Also
        --------
        :func:`pathlib.parent`
        """
        if isinstance(path, str):
            path = Path(path)
        return self.is_dir(path.parent)

    @abstractmethod
    def create_dir(self, path: Union[Path, str]) -> None:
        """
        Create a directory at a given path if it does not exist.

        Parameters
        ----------
        path: str or Path
            Full path to the directory to be created.
        """
