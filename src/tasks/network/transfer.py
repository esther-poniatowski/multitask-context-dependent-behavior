#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`transfer` [module]

Transfers files and directories between a local workspace and a remote server.

Functionalities:

- Transfer in both directions (upload and download).
- Organize the new directory structure.

Arguments
---------
--env-path : str
    Path to the `.env` file storing the remote server's credentials.
    See :meth:`load_network_config`.
--sync-map-path : str
    Path to the `sync-map.yml` file where the sync map is defined.
    See :meth:`load_sync_map`.
--direction : str
    Direction of the transfer: "upload" to send files to the remote server, "download" to retrieve
    files from the remote server.

Usage
-----
To upload files to the remote server:

.. code-block:: bash

    python transfer.py --env-path path/to/.env --sync-map-path path/to/sync_map.yml --direction upload

To download files from the remote server:

.. code-block:: bash

    python transfer.py --env-path path/to/.env --sync-map-path path/to/sync_map.yml --direction download

Notes
-----
Structure of the input sync map in the `sync_map.yml` file:

.. code-block:: yaml

     - source: path/to/source/directory/
       destination: path/to/destination/directory/
     - source: path/to/source/file.ext
       destination: path/to/destination/file.ext
     - source: path/to/source/file.ext
       destination: path/to/destination/directory/
     - source: path/to/source/file.ext
       destination: path/to/destination/directory/new_name.ext

Structure of the output sync map processed by the :class:`TransferManager` class:

.. code-block:: python

    [
        {"source": "path/to/source/directory/", "destination": "path/to/destination/directory/"},
        {"source": "path/to/source/file.ext", "destination": "path/to/destination/file.ext"},
        {"source": "path/to/source/file.ext", "destination": "path/to/destination/directory/"},
        {"source": "path/to/source/file.ext", "destination": "path/to/destination/directory/new_name.ext"}
    ]

Rules for the paths in the sync map:

- Paths are *relative* to the *root* directory of the workspace in the respective servers.
- To copy the *contents* of a directory, add a trailing slash to the source path.
- To copy the *directory itself*, do not add a trailing slash to the source path.
- To copy a single file, either specify its name in the destination (e.g. for renaming) or only
  specify its destination directory with a trailing slash.

"""
import argparse
from pathlib import Path
import subprocess
from typing import Dict, List, Union, Optional

from dotenv import dotenv_values

from utils.io_data.loaders.impl import LoaderYAML
from utils.io_data.formats import TargetType


class TransferManager:
    """
    Transfer files and directories to and from a remote server using a sync map.

    Class Attributes
    ----------------
    remote_cred : Dict[str, str]
        Mapping of the class attributes to the corresponding environment variables in the `.env`
        file containing the remote server's credentials.
    attr_types : Dict[str, type]
        Mapping of the class attributes to their respective types.

    Attributes
    ----------
    user : str
        Username on the remote server.
    host : str
        IP address or hostname of the remote server.
    root_path : Path
        Path of the root directory of the workspace on the remote server.
    sync_map : List of Dict[str, str]
        Mapping of local paths (files and directories) to the corresponding remote paths as
        specified in the `sync-map.yml` file.

    Methods
    -------
    :meth:`upload`
    :meth:`download`
    :meth:`ensure_dest_dir_exists`
    :meth:`load_sync_map`
    :meth:`load_network_config`

    See Also
    --------
    :class:`pathlib.Path`
        Object-oriented interface to the filesystem paths.
    :meth:`Path.resolve`:
        Resolve the full path of a file or directory, i.e., expand the path.
        Used here to get the full path of source files or directories.
    :meth:`subprocess.run`
        Execute a command in a subprocess.
    """

    remote_cred = {"user": "USER", "host": "HOST", "root_path": "ROOT"}
    attr_types = {"user": str, "host": str, "root_path": Path}

    def __init__(
        self,
        user: Optional[str] = None,
        host: Optional[str] = None,
        root_path: Optional[Union[Path, str]] = None,
        sync_map: Optional[List[Dict[str, str]]] = None,
    ):
        self.user = user
        self.host = host
        if root_path is not None:
            root_path = Path(root_path).resolve()  # absolute path
        self.root_path = root_path
        self.sync_map = sync_map if sync_map is not None else []  # avoid TypeError

    def upload(self, source_path, destination_path):
        """
        Upload files or directories to the remote server from the local machine.

        Arguments
        ---------
        source_path : str
            Path to the file or directory to upload on the local machine.
        destination_path : str
            Path to the destination file or directory on the remote server.
        """
        source_full_path = str(Path(source_path).resolve())
        destination_full_path = f"{self.user}@{self.host}:{self.root_path}/{destination_path}"
        self.ensure_remote_dir_exists(destination_full_path)
        self._run_rsync(source_full_path, destination_full_path)

    def download(self, source_path, destination_path):
        """
        Download files or directories from the remote server to the local machine.

        Arguments
        ---------
        source_path : str
            Path to the file or directory to download on the remote server.
        destination_path : str
            Path to the destination file or directory on the local machine.
        """
        source_full_path = f"{self.user}@{self.host}:{self.root_path}/{source_path}"
        destination_full_path = str(Path(destination_path).resolve())
        self.ensure_local_dir_exists(destination_full_path)
        self._run_rsync(source_full_path, destination_full_path)

    def ensure_remote_dir_exists(self, destination_path: str):
        """
        Ensure that a destination directory structure exists on the remote server.

        Arguments
        ---------
        destination_path : str
            Full path to a destination directory.

        See Also
        --------
        :command:`mkdir`
            Create a directory at the given path if it does not exist.
            Option `-p`: Create parent directories as needed.
        """
        directory = Path(destination_path).parent  # extract directory path
        command = ["ssh", f"{self.user}@{self.host}", f"mkdir -p {directory}"]
        subprocess.run(command, check=True)

    def ensure_local_dir_exists(self, destination_path: str):
        """
        Ensure that a destination directory structure exists on the local machine.

        Arguments
        ---------
        destination_path : str
            Full path to a destination directory.

        See Also
        --------
        :meth:`Path.mkdir`
            Create a directory at the given path if it does not exist.
            Option `parents=True`: Create parent directories as needed.
            Option `exist_ok=True`: Do not raise an error if the directory already exists.
        """
        directory = Path(destination_path).parent
        directory.mkdir(parents=True, exist_ok=True)

    def _run_rsync(self, source: str, destination: str):
        """
        Transfer one file or directory though remote synchronization.

        Arguments
        ---------
        source : str
            Path to the file or directory to transfer.
        destination : str
            Path to the destination file or directory.

        See Also
        --------
        :command:`rsync`
            Command-line utility to synchronize files and directories between two locations.
            Syntax (here): `rsync -avz <source_path> <destination_path>`

            Options:

            `-a` (archive)  : Preserve file attributes (e.g., timestamps, permissions...)
            `-v` (verbose)  : Display the progress of the transfer
            `-z`            : Compress data during the transfer
        """
        command = ["rsync", "-avz", source, destination]
        subprocess.run(command, check=True)

    def load_sync_map(self, path: Union[Path, str]):
        """
        Load the sync map from a YAML in the attribute :attr:`TransferManager.sync_map`.

        Arguments
        ---------
        path : str
            Path to the YAML file containing the sync map.

        See Also
        --------
        :class:`utils.io_data.loaders.impl.LoaderYAML`
        """
        loader = LoaderYAML(path, tpe=TargetType.DICT)
        self.sync_map = loader.load()

    def load_network_config(self, path: Union[Path, str]):
        """
        Extract user, host, and root path from a `.env` file and set corresponding attributes.

        Arguments
        ---------
        path : str
            Path to the `.env` file containing the network settings.

        Raises
        ------
        ValueError
            If the `.env` file does not contain the required network settings.

        See Also
        --------
        :func:`dotenv_values`
            Load environment variables from a .env file into a python dictionary, whose keys are the
            variable names and values are the values specified in the file.
        """
        path = Path(path).resolve()  # absolute path
        env_content = dotenv_values(path)
        connection_settings = {attr: env_content[var] for attr, var in self.remote_cred.items()}
        for key, value in connection_settings.items():
            if not value:
                raise ValueError(f"Missing network setting in .env file: {key}")
            setattr(self, key, self.attr_types[key](value))


def main():
    """Execute the file transfer process."""
    parser = argparse.ArgumentParser(description="Transfer files to and from a remote server.")
    parser.add_argument(
        "--env-path",
        type=str,  # expect a string path
        required=True,
        help="Path to the .env file storing the credentials for the remote server.",
    )
    parser.add_argument(
        "--sync-map-path",
        type=str,
        required=True,
        help="Path to the sync-map.yml file storing path correspondences local-remote.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["upload", "download"],
        required=True,
        help="Direction of transfer: 'upload' to send files to the server, 'download' to retrieve files from the server.",
    )
    args = parser.parse_args()

    transfer_manager = TransferManager()
    # Load configurations from files
    transfer_manager.load_network_config(args.env_path)
    transfer_manager.load_sync_map(args.sync_map_path)

    # Perform the transfer based on direction
    if args.direction == "upload":
        for paths in transfer_manager.sync_map:
            transfer_manager.upload(paths["source"], paths["destination"])
    elif args.direction == "download":
        for paths in transfer_manager.sync_map:
            transfer_manager.download(paths["source"], paths["destination"])


if __name__ == "__main__":
    main()
