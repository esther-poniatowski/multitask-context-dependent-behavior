#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`tasks.network.transfer` [module]

Transfers files and directories between a local workspace and a remote server.

Functionalities:

- Transfer in both directions: upload (from local to remote) and download (from remote to local).
- Process a sync map to transfer multiple files and directories at once.

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
--dry-run : flag
    Simulate the operations without executing them.

Usage
-----
To upload files to the remote server:

.. code-block:: bash

    python transfer.py --env-path path/to/.env --sync-map-path path/to/sync_map.yml --direction upload

To download files from the remote server:

.. code-block:: bash

    python transfer.py --env-path path/to/.env --sync-map-path path/to/sync_map.yml --direction download

To simulate an upload:

.. code-block:: bash

    python transfer.py --env-path path/to/.env --sync-map-path path/to/sync_map.yml --direction upload --dry-run


Notes
-----
Structure of the input sync map in the `sync_map.yml` file:

.. code-block:: yaml

     - source: path/to/source/directory/
       destination: path/to/destination/directory/
     - source: path/to/source/file.ext
       destination: path/to/destination/directory/
     - source: path/to/source/file.ext
       destination: path/to/destination/directory/new_name.ext

Structure of the output sync map processed by the :class:`TransferManager` class:

.. code-block:: python

    [
        {"source": "path/to/source/directory/", "destination": "path/to/destination/directory/"},
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

import yaml

from utils.io_data.loaders.impl import LoaderYAML
from utils.io_data.formats import TargetType
from utils.path_system.local_server import LocalServer
from utils.path_system.remote_server import RemoteServer


SyncMapType = List[Dict[str, Path]]
"""Type alias for the sync map: list of dictionaries with string keys and Path values. """


class TransferManager:
    """
    Transfer files and directories to and from a remote server using a sync map.

    Class Attributes
    ----------------
    valid_directions : List[str]
        List of valid directions for the transfer: "upload" and "download".

    Attributes
    ----------
    user : str
        Username on the remote server.
    host : str
        IP address or hostname of the remote server.
    root_remote : Path
        Custom path of the root directory of the workspace on the remote server.
    root_local : Path
        Custom path of the root directory of the workspace on the local machine.
    remote_server : :class:`RemoteServer`
        Remote server instance.
    local_server : :class:`LocalServer`
        Local server instance.
    sync_map : List of Dict[str, Path]
        Mapping of local paths (files and directories) to the corresponding remote paths as
        specified in the `sync-map.yml` file.
    direction : {"upload", "download"}
        Direction of the transfer: "upload" to send files to the remote server, "download" to
        retrieve files from the remote server.
    dry_run : bool, default=False
        If True, operations are only simulated rather than executed.

    Methods
    -------
    :meth:`transfer`
    :meth:`process_map`
    :meth:`_check_direction`
    :meth:`_run_rsync`
    :meth:`load_sync_map`

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

    valid_directions = ["upload", "download"]

    def __init__(
        self,
        user: Optional[str] = None,
        host: Optional[str] = None,
        root_remote: Optional[Union[Path, str]] = None,
        root_local: Optional[Union[Path, str]] = None,
        sync_map: Optional[SyncMapType] = None,
        direction: str = "upload",
        dry_run: bool = False,
    ):
        self.user = user
        self.host = host
        self.remote_server = RemoteServer(user=user, host=host, root_path=root_remote)
        self.local_server = LocalServer(root_path=root_local)
        self.sync_map = sync_map if sync_map is not None else []  # avoid TypeError
        self._check_direction(direction)
        self.direction = direction
        self.dry_run = dry_run

    def transfer(self, local_path: Path, remote_path: Path):
        """
        Upload or download one file or directory between the local machine and the remote server.

        Arguments
        ---------
        local_path : Path
            Path to the file or directory to transfer on the local machine.
        remote_path : Path
            Path to the destination file or directory to transfer on the remote server.
        """
        # Build full paths
        local_full_path = self.local_server.build_path(local_path)
        remote_full_path = self.remote_server.build_path(remote_path)
        # Set source and destination paths based on the transfer direction
        if self.direction == "upload":
            self.remote_server.is_dir(remote_full_path)
            source = str(local_full_path)
            destination = f"{self.user}@{self.host}:{remote_full_path}"
        elif self.direction == "download":
            self.local_server.is_dir(local_full_path)
            source = f"{self.user}@{self.host}:{remote_full_path}"
            destination = str(local_full_path)
        # Transfer files or directories
        print(f"[INFO] Transfer from {source} to {destination}")
        self._run_rsync(source, destination)

    def process_map(self):
        """
        Transfer all files or directories specified in the sync map.

        Arguments
        ---------
        direction : str, {"upload", "download"}
            Direction of the transfer: see :meth:`_check_direction`.

        See Also
        --------
        :meth:`transfer`
        """
        print(f"[INFO] Process sync map. Direction: {self.direction}.")
        for paths in self.sync_map:
            self.transfer(paths["source"], paths["destination"])

    def _check_direction(self, direction: str):
        """
        Check if the direction of the transfer is valid.

        Arguments
        ---------
        direction : {"upload", "download"}
            Direction of the transfer.

        Raises
        ------
        ValueError
            If the direction is not valid.
        """
        if direction not in self.valid_directions:
            raise ValueError(f"[ERROR] Invalid direction: {direction} ('upload'/'download')")

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
            `-n` (dry-run)  : Simulate the transfer without executing it

        Warning
        -------
        Paths should be strings to be passed to the subprocess command.
        """
        print(f"[INFO] Rsync from {source} to {destination}")
        command = ["rsync", "-avz"]
        if self.dry_run:
            command.append("--dry-run")
        command.extend([source, destination])
        subprocess.run(command, check=True)

    def load_sync_map(self, path: Union[Path, str]):
        """
        Load the sync map from a YAML in the attribute :attr:`TransferManager.sync_map`.

        Arguments
        ---------
        path : Union[Path, str]
            Path to the YAML file containing the sync map.

        Raises
        ------
        yaml.YAMLError
            If the YAML file cannot be loaded.
        ValueError
            If the sync map is not a list of dictionaries.

        See Also
        --------
        :class:`utils.io_data.loaders.impl.LoaderYAML`
        """
        path = Path(path).resolve()  # absolute path
        try:
            loader = LoaderYAML(path, tpe=TargetType.DICT)
            raw_map = loader.load()
            print(f"[SUCCESS] Load directory structure from YAML file at: {path}")
            if not isinstance(raw_map, list):  # ensure correct structure
                raise ValueError(f"[ERROR] Invalid type: {type(raw_map)} (expected List[Dict])")
            # If correct, set paths in the dictionary self.syn_map as Path objects
            self.sync_map = [
                {key: Path(value) for key, value in paths.items()} for paths in raw_map
            ]
        except yaml.YAMLError as exc:
            print(f"[ERROR] Failed loading YAML file: {exc}")
            raise exc


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
    parser.add_argument(
        "--dry-run",
        action="store_true",  # flag to set the dry_run attribute to True
        help="Simulate operations without executing them.",
    )
    args = parser.parse_args()

    # Initialize the transfer manager
    transfer_manager = TransferManager(direction=args.direction, dry_run=args.dry_run)
    # Load configurations from files
    transfer_manager.remote_server.load_network_config(args.env_path)
    transfer_manager.load_sync_map(args.sync_map_path)
    # Perform the transfer based on direction
    transfer_manager.process_map()


if __name__ == "__main__":
    main()
