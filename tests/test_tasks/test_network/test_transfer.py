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

import pytest

from tasks.network.transfer import TransferManager

# Relative path to the mock data directory based on the script's location
PATH_MOCK_DATA = Path(__file__).parent / "mock_data"


@pytest.fixture
def mock_credentials():
    """
    Fixture - Provide the credentials corresponding to the mock `.env` file.

    Returns
    -------
    dict
        Dictionary containing the user, host, and root path for the remote server.
    """
    return {
        "user": "test_user",
        "host": "111.111.1.1",
        "root_path": "/remote/root",
    }


def test_load_network_config(mock_credentials):
    """
    Test for :meth:`TransferManager.load_network_config`.

    Test Inputs
    -----------
    env_path : str
        Path to a mock `.env` file containing network credentials.
    mock_credentials : dict
        Dictionary containing the expected user, host, and root path, provided by the fixture,
        corresponding to the content of the `.env` file.

    Expected Output
    ---------------
    TransferManager attributes : `user`, `host`, `root_path` loaded from the `.env` file.
    """
    env_path = PATH_MOCK_DATA / ".env"
    transfer_manager = TransferManager()
    transfer_manager.load_network_config(env_path)
    assert transfer_manager.user == mock_credentials["user"], "User not loaded"
    assert transfer_manager.host == mock_credentials["host"], "Host not loaded"
    assert transfer_manager.root_path == Path(mock_credentials["root_path"]), "Root path not loaded"


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


def test_ensure_remote_dir_exists(mock_credentials, mocker):
    """
    Test for :meth:`TransferManager.ensure_remote_dir_exists`.

    Test Inputs
    -----------
    destination_path : str
        Full path to the directory on the remote server where the files will be uploaded.
        Here: `/remote/root/directory/file.txt`.
    mock_credentials : dict
        Dictionary containing the user, host, and root path for the remote server.

    Expected Output
    ---------------
    Check the execution of the appropriate SSH command to create the directory.

    Implementation
    --------------
    Mock the `subprocess.run` method to check the execution of the SSH command.
    """
    mock_run = mocker.patch("subprocess.run")
    transfer_manager = TransferManager(**mock_credentials)
    destination_path = "/remote/root/directory/file.txt"
    expected_dir = "/remote/root/directory"
    transfer_manager.ensure_remote_dir_exists(destination_path)
    mock_run.assert_called_once_with(
        [
            "ssh",
            f"{mock_credentials['user']}@{mock_credentials['host']}",
            f"mkdir -p {expected_dir}",
        ],
        check=True,
    )


def test_ensure_local_dir_exists(tmp_path):
    """
    Test for :meth:`TransferManager.ensure_local_dir_exists`.

    Test Inputs
    -----------
    destination_path : str
        Full path to the directory on the local machine where the files will be downloaded.

    Expected Output
    ---------------
    Check that the directory structure is created locally.
    """
    transfer_manager = TransferManager()
    destination_path = "directory/file.txt"
    expected_dir = tmp_path / "directory"
    local_path = tmp_path / destination_path
    transfer_manager.ensure_local_dir_exists(str(local_path))
    assert expected_dir.exists(), "Local directory not created"


def test_run_rsync(mocker):
    """
    Test for :meth:`TransferManager._run_rsync`.

    Test Inputs
    -----------
    source : str
        Path to the file or directory to transfer.
    destination : str
        Path to the destination file or directory.

    Expected Output
    ---------------
    Check that the rsync command is called with the correct arguments.

    Implementation
    --------------
    Mock the `subprocess.run` method to check the execution of the rsync command.
    """
    mock_run = mocker.patch("subprocess.run")
    transfer_manager = TransferManager()
    source = "/source/path/file.txt"
    destination = "/destination/path/file.txt"
    transfer_manager._run_rsync(source, destination)
    mock_run.assert_called_once_with(["rsync", "-avz", source, destination], check=True)
