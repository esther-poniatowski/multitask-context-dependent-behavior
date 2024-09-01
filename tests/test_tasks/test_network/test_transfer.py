#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.test_transfer` [module]

Test the file transfer functionality of the `TransferManager` class.

Implementation
--------------
Mock data for the `.env` file and the sync map: `tests/test_tasks/test_network/mock_data`.

See Also
--------
:mod:`transfer`: Tested module.
:mod:`socket`: Used to get the IP address of the local machine (standard library).
"""
from pathlib import Path
import socket

import pytest

from tasks.network.transfer import TransferManager

# Relative path to the mock data directory based on the script's location
PATH_MOCK_DATA = Path(__file__).parent / "mock_data"


def test_load_network_config():
    """
    Test for :meth:`TransferManager.load_network_config`.

    Test Inputs
    -----------
    env_path : str
        Path to a mock `.env` file containing network credentials.

    Expected Output
    ---------------
    TransferManager attributes : `user`, `host`, `root_path` loaded from the `.env` file.
    """
    env_path = PATH_MOCK_DATA / ".env"
    transfer_manager = TransferManager()
    transfer_manager.load_network_config(env_path)
    assert transfer_manager.user == "test_user", "User not loaded correctly"
    assert transfer_manager.host == "111.111.1.1", "Host not loaded correctly"
    assert isinstance(transfer_manager.root_path, Path), "Root path not loaded correctly"


def test_load_sync_map():
    """
    Test for :meth:`TransferManager.load_sync_map`.

    Test Inputs
    -----------
    sync_map_path : str
        Path to a mock YAML file containing the sync map.

    Expected Output
    ---------------
    TransferManager attribute : `sync_map` as a list of dictionaries.
    Keys in each dictionary : `source`, `destination`.
    """
    sync_map_path = PATH_MOCK_DATA / "sync_map.yml"
    transfer_manager = TransferManager()
    transfer_manager.load_sync_map(sync_map_path)
    # Check sync map structure
    assert isinstance(transfer_manager.sync_map, list), "Sync map not loaded as a list"
    assert all(
        isinstance(item, dict) for item in transfer_manager.sync_map
    ), "Sync map not containing dictionaries"
    assert all(
        {"source", "destination"}.issubset(item.keys()) for item in transfer_manager.sync_map
    ), "Sync map dictionaries missing keys"


@pytest.mark.skip(reason="Not working for a local transfer.")
def test_transfer(tmp_path):
    """
    Test for :meth:`TransferManager.upload`.

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
    # Get the current user's name and IP address of the local machine
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    # Set paths for the source and destination files
    source_path = PATH_MOCK_DATA / "file.txt"
    destination_path = "directory/file.txt"
    # Create a TransferManager instance and transfer the file
    transfer_manager = TransferManager(user=hostname, host=ip_address, root_path=tmp_path)
    transfer_manager.upload(source_path, destination_path)
    # Check that the file was transferred to the correct location
    assert (tmp_path / destination_path).exists(), "File not transferred correctly"
