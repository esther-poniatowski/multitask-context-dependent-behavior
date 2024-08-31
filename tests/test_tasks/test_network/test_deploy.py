#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.test_deployer` [module]

Test the deployment functionality of the `Deployer` class.

Notes
-----
This module tests the loading of environment variables from a `.env` file, the parsing of a sync map
from a YAML file, and the correct execution of file transfers using `rsync`.

Implementation
--------------
The tests use mock data for the `.env` file and the sync map, and the `subprocess.run` command is
mocked to avoid actual file transfers during testing.

See Also
--------
:mod:`deploy`: Tested module.
:mod:`socket`: Used to get the IP address of the local machine (standard library).
"""
from pathlib import Path
import socket

import pytest

from tasks.network.deploy import Deployer

# Relative path to the mock data directory based on the script's location
PATH_MOCK_DATA = Path(__file__).parent / "mock_data"


def test_load_network_config():
    """
    Test for :meth:`Deployer.load_network_config`.

    Test Inputs
    -----------
    env_path : str
        Path to a mock `.env` file containing network credentials.

    Expected Output
    ---------------
    Deployer attributes : `user`, `host`, `root_path` loaded from the `.env` file.
    """
    env_path = PATH_MOCK_DATA / ".env"
    deployer = Deployer()
    deployer.load_network_config(env_path)
    assert deployer.user is not None, "User not loaded"
    assert deployer.host is not None, "Host not loaded"
    assert isinstance(deployer.root_path, Path), "Root path not loaded"


def test_load_sync_map():
    """
    Test for :meth:`Deployer.load_sync_map`.

    Test Inputs
    -----------
    sync_map_path : str
        Path to a mock YAML file containing the sync map.

    Expected Output
    ---------------
    Deployer attribute : `sync_map` as a list of dictionaries.
    Keys in each dictionary : `source`, `destination`.
    """
    sync_map_path = PATH_MOCK_DATA / "sync_map.yml"
    deployer = Deployer()
    deployer.load_sync_map(sync_map_path)
    # Check sync map structure
    assert isinstance(deployer.sync_map, list), "Sync map not loaded as a list"
    assert all(
        isinstance(item, dict) for item in deployer.sync_map
    ), "Sync map not containing dictionaries"
    assert all(
        {"source", "destination"}.issubset(item.keys()) for item in deployer.sync_map
    ), "Sync map dictionaries missing keys"


@pytest.mark.skip(reason="Not working for a local transfer.")
def test_transfer(tmp_path):
    """
    Test for :meth:`Deployer.transfer`.

    Test Inputs
    -----------
    source_path : str
        Path to the source file or directory to transfer.
    destination_path : str
        Path to the destination on the remote server.

    Expected Output
    ---------------
    Check that the file is transferred to the correct location on the simulated remote server.

    Implementation
    --------------
    Here, a local transfer is simulated by using the local machine's IP address as the remote host.
    Source file: `file.txt` in the mock data directory.
    Destination path: `directory/file.txt` on the remote server.
    Root path on the remote server: Temporary directory created by `pytest`.
    """
    # Get the current user's name and ip address of the local machine
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    # Set paths for the source and destination files
    source_path = PATH_MOCK_DATA / "file.txt"
    destination_path = "directory/file.txt"
    # Create a Deployer instance and transfer the file
    deployer = Deployer(user=hostname, host=ip_address, root_path=tmp_path)
    deployer.transfer(source_path, destination_path)
    # Check that the file was transferred to the correct location
    assert (tmp_path / destination_path).exists(), "File not transferred"
