#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`tasks.network.setup_directories` [module]

Organize directory structures on a server (local or remote) based on a YAML configuration.

Functionalities:
- Create directories as specified in the YAML file.
- Ensure the full directory structure exists on the server.

Arguments
---------
--yml-dirstruct-path : str
    Path to the YAML file defining the directory structure.
--env-path : str, optional
    Path to the `.env` file storing the remote server's credentials.
    If not provided, the directory structure is organized locally.
--dry-run : flag
    Simulate the operations without executing them.

Warning
-------
If this script is executed locally, ensure that either one of the following conditions is met:

- The `ROOT` environment variable is set in the current shell session.
- The script is run in the root directory of the project.

Usage
-----
To organize the directory structure locally:

.. code-block:: bash

    python setup_directories.py --yml-dirstruct-path path/to/structure.yml

To organize the directory structure on a remote server:

.. code-block:: bash

    python setup_directories.py --yml-dirstruct-path path/to/structure.yml --env-path path/to/.env

To simulate the directory organization:

.. code-block:: bash

    python setup_directories.py --yml-dirstruct-path path/to/structure.yml --dry-run

Notes
-----
Directory tree are specified by nested dictionaries in the YAML file and python objects. Each key
represents a directory name. Each value can be another dictionary (indicating subdirectories) or an
empty dictionary if the directory has no subdirectories.

Example directory structure:

.. code-block:: plaintext

    root/
    └── dir/
        ├── subdir1/
        └── subdir2/
            ├── subsubdir1/
            └── subsubdir2/

Format of the YAML file (relative to the root directory, which is implicit):

.. code-block:: yaml

    dir:
      subdir1: {}
      subdir2:
        subsubdir1: {}
        subsubdir2: {}

Format of the python object after loading the YAML:

.. code-block:: python

    {
        "dir": {
            "subdir1": {},
            "subdir2": {
                "subsubdir1": {},
                "subsubdir2": {}
            }
        }
    }

Classes
-------
:class:`DirectoryOrganizer`
"""

import argparse
from pathlib import Path
from typing import Dict, Union, Optional

import yaml

from utils.path_system.local_server import LocalServer
from utils.path_system.remote_server import RemoteServer
from utils.io_data.loaders.impl_loaders import LoaderYAML
from utils.io_data.formats import TargetType


StructureType = Dict[str, Union[Dict, str]]
"""Type alias for the directory structure: Nested dictionary representing the directory tree."""


class DirectoryOrganizer:
    """
    Organize directory structures on a server (local or remote) based on a YAML file.

    Parameters
    ----------
    remote : bool, default=False
        Whether the directories are being organized on a remote server.
    root_path : Optional[Path], optional
        Custom root path for the directory structure to provide to the server.

    Attributes
    ----------
    server : Union[LocalServer, RemoteServer]
        Server instance to organize the directory structure.
    directory_structure : Dict[str, Union[Dict, List[str]]], optional
        Directory structure to organize as specified in the YAML file.
    dry_run : bool, default=False
        If True, operations are only simulated rather than executed.

    Methods
    -------
    :meth:`get_root_local` (static method)
    :meth:`create_directories`
    :meth:`check_dir_exists`
    :meth:`load_directory_structure`
    """

    def __init__(
        self,
        remote: bool = False,
        root_path: Optional[Path] = None,
        directory_structure: Optional[StructureType] = None,
        dry_run: bool = False,
    ):
        if remote:
            self.server = RemoteServer(root_path=root_path)
        else:
            self.server = LocalServer(root_path=root_path)
        if not hasattr(self.server, "root_path"):
            raise ValueError("[ERROR] Attribute `root_path` not set.")
        self.dry_run = dry_run
        self.directory_structure = directory_structure if directory_structure is not None else {}

    def create_directories(
        self, current_path: Optional[Path] = None, structure: Optional[StructureType] = None
    ):
        """
        Recursively create directories based on the provided structure.

        Arguments
        ---------
        current_path : Path
            Current root path for directory creation.
            If not provided, the attribute :attr:`root_path` is used.
        structure : Dict[str, Union[Dict, str]]
            Directory structure to create.
            If not provided, the instance attribute :attr:`directory_structure` is used.

        Notes
        -----
        This method can be called either:

        - Without arguments to create the directories based on the instance attributes.
        - With arguments to create directories based on a specific structure and path.

        Implementation
        --------------
        - For each key in the structure, create a directory at the current path.
        - If the value is a dictionary, recursively create subdirectories by calling this method
          again with the new path and substructure. New path: current path + key. New structure:
          value of the current key (i.e., substructure).
        - If the directory already exists, skip the creation and continue with the next directory.
        """
        if current_path is None:
            current_path = self.server.root_path
        if structure is None:
            structure = self.directory_structure
        for new_dir, new_struct in structure.items():
            dir_path = current_path / new_dir
            if not self.server.is_dir(dir_path):
                try:
                    if not self.dry_run:
                        self.server.create_dir(dir_path)
                    else:
                        print(f"[DRY-RUN] Would create directory: {dir_path}")
                except Exception as exc:
                    print(f"[ERROR] Failed to create directory {dir_path}: {exc}")
            if isinstance(new_struct, dict):  # recursively create subdirectories
                self.create_directories(dir_path, new_struct)

    def load_directory_structure(self, path: Union[Path, str]):
        """
        Load the directory structure from a YAML file.

        Arguments
        ---------
        path : Union[Path, str]
            Path to the YAML file containing the directory structure.

        Raises
        ------
        yaml.YAMLError
            If an error occurs while loading the YAML file.

        See Also
        --------
        :class:`utils.io_data.loaders.impl_loaders.LoaderYAML`
        """
        path = Path(path).resolve()  # absolute path
        try:
            loader = LoaderYAML(path, tpe=TargetType.DICT)
            directory_structure = loader.load()
            print(f"[SUCCESS] Load directory structure from YAML file at: {path}")
            if not isinstance(directory_structure, dict):  # ensure correct structure
                raise ValueError("[ERROR] Directory structure must be a dictionary.")
            self.directory_structure = directory_structure  # update instance attribute if correct
        except yaml.YAMLError as exc:
            print(f"[ERROR] Failed loading YAML file: {exc}")
            raise exc


def main():
    """Execute the directory organization process."""
    parser = argparse.ArgumentParser(
        description="Organize directory structure on a local or remote server."
    )
    parser.add_argument(
        "--yml-dirstruct-path",
        type=str,
        required=True,
        help="Path to the YAML file defining the directory structure.",
    )
    parser.add_argument(
        "--env-path",
        type=str,
        required=False,
        help="Path to the .env file storing the credentials for the remote server. Omit to run locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the operations without executing them.",
    )
    args = parser.parse_args()
    # Set the remote flag based on the presence of the env-path argument
    remote = args.env_path is not None

    # Initialize the directory organizer
    organizer = DirectoryOrganizer(remote=remote, dry_run=args.dry_run)
    # Load remote network configuration if env-path is provided
    if remote:
        organizer.server.load_network_config(args.env_path)
    # Load the directory structure from the YAML file
    organizer.load_directory_structure(args.yml_dirstruct_path)
    # Create the directory structure (locally or remotely)
    organizer.create_directories()


if __name__ == "__main__":
    main()
