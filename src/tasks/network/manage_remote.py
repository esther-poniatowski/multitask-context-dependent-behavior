#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`manage_remote` [module]

Manage interactions with a remote server.

Functionalities:
- Load remote server credentials from `.env` files.
- Check the existence of directories on a remote server.

Classes
-------
:class:`RemoteServerMixin`
"""
import subprocess
from pathlib import Path
from typing import Union, Optional
from dotenv import dotenv_values


class RemoteServerMixin:
    """
    Mixin class for managing interactions with a remote server.

    Class Attributes
    ----------------
    remote_cred : Dict[str, str]
        Mapping of the class attributes to the corresponding environment variables in the `.env`
        file containing the remote server's credentials.
    attr_types : Dict[str, type]
        Mapping of the class attributes to their respective types.
    default_root : Path
        Default root directory path on the remote server: name of the project in the user's home.

    Attributes
    ----------
    user : str
        Username on the remote server.
    host : str
        IP address or hostname of the remote server.
    root_remote : Path
        Root directory path on the remote server.

    Methods
    -------
    :meth:`load_network_config`
    :meth:`build_path_remote`
    :meth:`is_dir_remote`
    :meth:`create_dir_remote`

    Notes
    -----
    For method names, see the corresponding local operations in :mod:`utils.path_system.explorer`.
    Add the suffix `_remote` to the method names to distinguish them from local operations when the
    mixin class is inherited by a class which also manages local directories.
    """

    remote_cred = {"user": "USER", "host": "HOST", "root_remote": "ROOT"}
    attr_types = {"user": str, "host": str, "root_remote": Path}
    default_root = Path("~/mtcdb").expanduser()

    def __init__(
        self,
        user: Optional[str] = None,
        host: Optional[str] = None,
        root_remote: Union[Path, str] = default_root,
    ):
        self.user = user
        self.host = host
        self.root_remote = Path(root_remote).resolve()

    def load_network_config(self, path: Union[Path, str]) -> None:
        """
        Extract user, host, and root path from a `.env` file.

        Arguments
        ---------
        path : Union[Path, str]
            Path to the `.env` file containing the network settings.

        Raises
        ------
        ValueError
            If the `.env` file does not contain the required network settings.
        """
        path = Path(path).resolve()
        print("Loading network settings from .env file at:", path)
        env_content = dotenv_values(path)
        connection_settings = {attr: env_content[var] for attr, var in self.remote_cred.items()}
        for key, value in connection_settings.items():
            if not value:
                raise ValueError(f"Missing network setting in .env file: {key}")
            setattr(self, key, self.attr_types[key](value))
        print(f"Credentials for the remote server: {self.user}@{self.host}")
        print(f"Root path: {self.root_remote}")

    def build_path_remote(self, path: Path) -> Path:
        """
        Build a full path on the remote server from the root path.

        Arguments
        ---------
        path : Path
            Relative path on the remote server, from the root directory of the workspace.

        Returns
        -------
        full_path: Path
            Full path on the remote server, encompassing the root path.
        """
        return self.root_remote / path

    def is_dir_remote(self, path: Path) -> bool:
        """
        Check if a directory exists on the remote server.

        Arguments
        ---------
        path : Path
            Full path to the directory on the remote server.

        Returns
        -------
        exists : bool
            True if the directory exists, False otherwise.

        See Also
        --------
        :command:`test`
            Check file types and compare values.
            Option `-d`: Check if the path is a directory.
        """
        command = ["ssh", f"{self.user}@{self.host}", f"test -d {path}"]
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            print(f"[VALID] Existing directory: {path} on {self.host}")
            return True
        else:
            print(f"[WARNING] Missing directory: {path} on {self.host}")
            return False

    def create_dir_remote(self, path):
        """
        Create a directory on the remote server.

        Arguments
        ---------
        path : Path
            Full path to the directory on the remote server.

        See Also
        --------
        :command:`mkdir -p`
            Create directories.
            Option `-p`(`--parents`): Create parent directories if needed.
        """
        command = ["ssh", f"{self.user}@{self.host}", f"mkdir -p {path}"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"[ERROR] {result.stderr}")
        else:
            print(f"[SUCCESS] Directory created: {path} on {self.host}")
