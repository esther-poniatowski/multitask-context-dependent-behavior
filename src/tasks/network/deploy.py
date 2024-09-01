#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`deploy` [module]

Deploy a part of the workspace to a remote server.

Transfer a set of files and directories and organize the new directory structure.

Usage
-----
.. code-block:: bash

    python deploy.py --env-path path/to/.env --sync-map-path path/to/sync-map.yml

Arguments
---------
--env-path : str
    Path to the `.env` file storing the network credentials (see :meth:`load_network_config`).
--sync-map-path : str
    Path to the `sync-map.yml` file where the sync map is defined (see :meth:`load_sync_map`).

Notes
-----
Structure of the input sync map in the `sync-map.yml` file:

.. code-block:: yaml

     - source: path/to/source/directory/
       destination: path/to/destination/directory/
     - source: path/to/source/file.ext
       destination: path/to/destination/file.ext
     - source: path/to/source/file.ext
       destination: path/to/destination/directory/
     - source: path/to/source/file.ext
       destination: path/to/destination/directory/new_name.ext


Structure of the output sync map processed by the :class:`Deployer` class:

.. code-block:: python

    [
        {"source": "path/to/source/directory/", "destination": "path/to/destination/directory/"},
        {"source": "path/to/source/file.ext", "destination": "path/to/destination/file.ext"},
        {"source": "path/to/source/file.ext", "destination": "path/to/destination/directory/"},
        {"source": "path/to/source/file.ext", "destination": "path/to/destination/directory/new_name.ext"}
    ]

"""
import argparse
from pathlib import Path
import subprocess
from typing import Dict, List, Union, Optional

from dotenv import dotenv_values

from utils.io_data.loaders.impl import LoaderYAML
from utils.io_data.formats import TargetType


class Deployer:
    """
    Deploy files to a remote server using a sync map.

    Class Attributes
    ----------------
    env_keys : Dict[str, str]
        Mapping of the class attributes to the corresponding environment variables in the `.env`
        file.
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
        Structure: List of dictionaries, each representing a pair of source and destination paths.
        Each dictionary holds two keys: `source` and `destination`, to specify the respective paths.

    Methods
    -------
    :meth:`deploy`
    :meth:`transfer`
    :meth:`load_sync_map`
    :meth:`load_network_config`

    See Also
    --------
    :class:`pathlib.Path`
    :mod:`subprocess`
    """

    env_keys = {"user": "USER", "host": "HOST", "root_path": "ROOT"}
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

    def deploy(self):
        """Transfer files and directories to the remote server based on the sync map."""
        for paths in self.sync_map:
            self.transfer(paths["source"], paths["destination"])

    def transfer(self, source_path, destination_path):
        """
        Transfer one file or directory to the remote server though remote synchronization.

        Arguments
        ---------
        source_path : str
            Path to the file or directory to transfer.
        destination_path : str
            Path to the destination file or directory on the remote server.

        See Also
        --------
        :meth:`Path.resolve`:
            Resolve the full path of a file or directory, i.e., expand the path.
            Used here to get the full path of the source file or directory.
        :meth:`subprocess.run`
            Execute a command in a subprocess.
            Used here to run the rsync command.
        :command:`rsync`
            Command-line utility to synchronize files and directories between two locations.
            Syntax (here): `rsync -avz <source_path> <destination_path>`

            Options:

            `-a` (archive)  : Preserve file attributes (e.g., timestamps, permissions...)
            `-v` (verbose)  : Display the progress of the transfer
            `-z`            : Compress data during the transfer
        """
        source_full_path = str(Path(source_path).resolve())
        destination_full_path = f"{self.user}@{self.host}:{self.root_path}/{destination_path}"
        command = ["rsync", "-avz", source_full_path, destination_full_path]
        subprocess.run(command, check=True)

    def load_sync_map(self, path: Union[Path, str]):
        """
        Load the sync map from a YAML in the attribute :attr:`Deployer.sync_map`.

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
        connection_settings = {attr: env_content[var] for attr, var in self.env_keys.items()}
        for key, value in connection_settings.items():
            if not value:
                raise ValueError(f"Missing network setting in .env file: {key}")
            setattr(self, key, self.attr_types[key](value))


def main():
    """Execute the deployment process."""
    parser = argparse.ArgumentParser(description="Deploy files to a remote server.")
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
    args = parser.parse_args()
    deployer = Deployer()
    # Load configurations from files
    deployer.load_network_config(args.env_path)
    deployer.load_sync_map(args.sync_map_path)
    # Deploy files to the remote server
    deployer.deploy()


if __name__ == "__main__":
    main()
