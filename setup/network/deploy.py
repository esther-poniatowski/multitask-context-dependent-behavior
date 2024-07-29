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
    Path to the `.env` file where the network credentials are defined (see :func:`load_network_config`).
--sync-map-path : str
    Path to the `sync-map.yml` file where the sync map is defined (see :attrs:`Deployer.sync_map`).

Example
-------
.. code-block:: bash

    python setup/manage_env_vars.py --env-path .env

Notes
-----
Structure of the input sync map:

.. code-block:: yaml

    directories:
        - local: path/to/local/directory1
        remote: path/to/remote/directory1
        - local: path/to/local/directory2
        remote: path/to/remote/directory2
    files:
        - local: path/to/local/file1
        remote: path/to/remote/file1
        - local: path/to/local/file2
        remote: path/to/remote/file2


Structure of the output sync map:

.. code-block:: python

    {
        'directories': [
            {'local': 'path/to/local/directory1', 'remote': 'path/to/remote/directory1'},
            {'local': 'path/to/local/directory2', 'remote': 'path/to/remote/directory2'}
        ],
        'files': [
            {'local': 'path/to/local/file1', 'remote': 'path/to/remote/file1'},
            {'local': 'path/to/local/file2', 'remote': 'path/to/remote/file2'}
        ]
    }

"""
import argparse
import os
from pathlib import Path
import subprocess
from typing import Dict, Union, Optional

from dotenv import load_dotenv
import yaml


class Deployer:
    """
    Deploy files to a remote server.

    Attributes
    ----------
    user : str
        Username on the remote server.
    host : str
        IP address or hostname of the remote server.
    root_path : Path
        Path of the root directory of the workspace on the remote server.
    sync_map : dict
        Mapping of local paths (files and directories) to the corresponding remote paths.
        The structure and content of the directory corresponds to the `sync-map.yml` file.
        Two keys: `directories` and `files`, each containing a list of dictionaries.
        Within each dictionary, two keys `local` and `remote` specify the respective paths.

    Methods
    -------
    :meth:`deploy`
    :meth:`rsync_transfer`

    See Also
    --------
    :class:`pathlib.Path`
    :mod:`subprocess`
    """
    def __init__(self,
                 user: str,
                 host:str,
                 root_path: Union[Path, str],
                 sync_map: Dict[str, str]
    ):
        self.user = user
        self.host = host
        self.root_path = Path(root_path)
        self.sync_map = sync_map

    def deploy(self):
        """Process the sync map to transfer files and directories to the remote server."""
        for key in ['directories', 'files']:
            for map in self.sync_map.get(key, []):
                self.rsync_transfer(map['local'], map['remote'])

    def rsync_transfer(self, local_path, path):
        """
        Transfer a file or directory to the remote server using rsync.

        Notes
        -----
        Remote synchronization is performed using the `rsync` command.

        Syntax:

        .. code-block:: bash

                rsync -avz <local_path> <remote_path>

        Options:

            -a: Archive mode - Preserve file attributes (e.g., timestamps, permissions...)
            -v: Verbose mode - Display the progress of the transfer
            -z: Compress data during the transfer

        Raises
        ------
        RuntimeError
            If the rsync command fails.

        See Also
        --------
        :meth:`Path.resolve`:
            Resolve the full path of a file or directory, i.e., expand the path.
            Used here to get the full path of the local file or directory.
        :meth:`subprocess.run`
            Execute a command in a subprocess.
            Used here to run the rsync command.
        """
        local_full_path = Path(local_path).resolve()
        full_path = f"{self.user}@{self.host}:{self.root_path}/{path}"
        command = ['rsync', '-avz', str(local_full_path), full_path]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully synced {local_full_path} to {full_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to sync {local_full_path} to {full_path}") from e


def load_network_config(path: Union[Path, str]) -> Dict[str, str]:
    """
    Load network settings from a `.env` file and extract the user, host, and root path.

    Arguments
    ---------
    path : str
        Path to the `.env` file containing the network settings.

    Returns
    -------
    connection_settings : dict
        Dictionary containing the user, host, and root path.

    Raises
    ------
    ValueError
        If the `.env` file does not contain the required network settings.

    See Also
    --------
    :func:`load_dotenv`
        Load environment variables from a file in a dictionary of the variables.
    """
    load_dotenv(path)
    connection_settings = {
        'user': os.getenv('USER', ''),
        'host': os.getenv('HOST', ''),
        'root_path': os.getenv('ROOT', '')
    }
    missing = [key for key, value in connection_settings.items() if not value]
    if missing:
        raise ValueError("Missing network settings in .env file: {}".format(', '.join(missing)))
    return connection_settings


def load_sync_map(path):
    """
    Load the sync map from a YAML.

    Arguments
    ---------
    path : str
        Path to the YAML file containing the sync map (see :attr:`Deployer.sync_map`).

    Returns
    -------
    dict
        Dictionary containing the sync map.

    See Also
    --------
    :func:`yaml.safe_load`
        Parse a YAML file and return the corresponding Python object.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deploy files to a remote server.")
    parser.add_argument('--env-path', type=str, required=True, help='Path to the .env file.')
    parser.add_argument('--sync-map-path', type=str, required=True, help='Path to the sync-map.yml file.')
    args = parser.parse_args()

    network_config = load_network_config(args.env_path)
    sync_map = load_sync_map(args.sync_map_path)

    deployer = Deployer(
        user=network_config['user'],
        host=network_config['host'],
        root_path=network_config['root_path'],
        sync_map=sync_map
    )
    deployer.deploy()
