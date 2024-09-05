#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`setup_directories` [module]

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
Directory tree are specified by nested dictionaries in the YAML file and python objects.
Each key represents a directory name. Each value can be another dictionary (indicating
subdirectories) or an empty dictionary if the directory has no subdirectories.

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
import os
from pathlib import Path
from typing import Dict, Union, Optional

import yaml

from tasks.network.manage_remote import RemoteServerMixin
from utils.path_system.explorer import is_dir, create_dir

# Type alias for the directory structure
StructureType = Dict[str, Union[Dict, str]]


class DirectoryOrganizer(RemoteServerMixin):
    """
    Organize directory structures on a server (local or remote) based on a YAML file.

    Attributes
    ----------
    root_path : Path, optional
        Root path for directory organization on the server.
        If not provided and :attr:`remote` is True, it is set when the mixin class is initialized.
        If not provided and :attr:`remote` is False, see :meth:`get_root_path`.
    directory_structure : Dict[str, Union[Dict, List[str]]], optional
        Directory structure to organize as specified in the YAML file.
    dry_run : bool, default=False
        If True, operations are only simulated rather than executed.
    remote : bool, default=False
        Whether the directories are being organized on a remote server.

    Methods
    -------
    :meth:`get_root_path` (static method)
    :meth:`create_directories`
    :meth:`check_dir_exists`
    :meth:`load_directory_structure`
    """

    def __init__(
        self,
        root_path: Optional[Path] = None,
        directory_structure: Optional[StructureType] = None,
        dry_run: bool = False,
        remote: bool = False,
    ):
        super().__init__()  # initialize RemoteServerMixin
        self.remote = remote
        self.dry_run = dry_run
        if root_path is not None:  # set root_path manually if provided
            self.root_path = root_path
        if root_path is None:
            if not remote:  # local server: set root_path from environment variable (or cwd)
                self.root_path = self.get_root_path()
            else:  # remote server: check if root_path is set by RemoteServerMixin
                if not hasattr(self, "root_path"):
                    raise ValueError("[ERROR] Attribute `root_path` not set.")
        self.directory_structure = directory_structure if directory_structure is not None else {}

    @staticmethod
    def get_root_path() -> Path:
        """
        Get the root path for directory organization on the local server.

        - If the `ROOT` environment variable is set, it is used as the root path.
        - Otherwise, the root path defaults to the current working directory.

        Returns
        -------
        Path
            Root path for directory organization.
        """
        root_value = os.environ.get("ROOT")
        return Path(root_value) if root_value is not None else Path.cwd()

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
            current_path = self.root_path
        if structure is None:
            structure = self.directory_structure
        for key, value in structure.items():
            dir_path = current_path / key
            if not self.check_dir_exists(dir_path):
                try:
                    if not self.dry_run:
                        if self.remote:
                            self.create_dir_remote(dir_path)  # method from RemoteServerMixin
                        else:
                            create_dir(dir_path)  # utility function from utils.path_system.explorer
                    else:  # simulate directory creation
                        print(f"[DRY-RUN] Would create directory: {dir_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to create directory {dir_path}: {e}")
            if isinstance(value, dict):  # recursively create subdirectories
                self.create_directories(dir_path, value)

    def check_dir_exists(self, path: Path):
        """
        Determines whether a directory exists on the server.

        Run the appropriate method based on the server type (local or remote).

        Arguments
        ---------
        path : Path
            Full path to a directory on the server.

        Returns
        -------
        bool
            True if the directory exists, False otherwise.
        """
        if self.remote:
            return self.is_dir_remote(path)  # method from RemoteServerMixin
        else:
            return is_dir(path)  # utility function from utils.path_system.explorer

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
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                print(f"[SUCCESS] Load directory structure from YAML file at: {path}")
                self.directory_structure = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"[ERROR] Failed loading YAML file: {e}")
            raise e


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
    organizer = DirectoryOrganizer(dry_run=args.dry_run, remote=remote)
    # Load remote network configuration if env-path is provided
    if remote:
        organizer.load_network_config(args.env_path)
    # Load the directory structure from the YAML file
    organizer.load_directory_structure(args.structure_path)
    # Create the directory structure (locally or remotely)
    organizer.create_directories()


if __name__ == "__main__":
    main()
