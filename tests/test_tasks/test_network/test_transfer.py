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
        "root_path": Path("/remote/root"),
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
    transfer_manager = TransferManager(dry_run=False)
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
    transfer_manager = TransferManager(dry_run=False)
    transfer_manager.load_sync_map(sync_map_path)
    # Check sync map structure
    assert isinstance(transfer_manager.sync_map, list), "Sync map not loaded as a list"
    assert all(
        isinstance(item, dict) for item in transfer_manager.sync_map
    ), "Sync map not containing dictionaries"
    assert all(
        {"source", "destination"}.issubset(item.keys()) for item in transfer_manager.sync_map
    ), "Sync map dictionaries missing keys"


@pytest.mark.parametrize("dry_run", argvalues=[True, False], ids=["dry_run", "normal"])
def test_ensure_remote_dir_exists(mock_credentials, mocker, dry_run):
    """
    Test for :meth:`TransferManager.ensure_remote_dir_exists`.

    Test Inputs
    -----------
    destination_path : str
        Full path to the directory on the remote server where the files will be uploaded.
        Here: `/remote/root/directory/file.txt`.
    mock_credentials : dict
        Dictionary containing the user, host, and root path for the remote server.
    dry_run : bool
        Whether the test should simulate the operations (dry_run=True) or perform them (dry_run=False).

    Expected Output
    ---------------
    Execution of the appropriate SSH commands to first check if the directory exists, and then
    create it if not in dry-run mode.

    Implementation
    --------------
    1. Simulate the scenario where the directory does not exist.
       `side_effect`: List of return values for each call to the mocked method.
       First value: `returncode=1` to simulate the directory not existing.
       Second value: `returncode=0` to simulate the successful creation of the directory.
    2. Mock the `subprocess.run` method to check the execution of the SSH commands.
    """
    mock_run = mocker.patch("subprocess.run")
    transfer_manager = TransferManager(**mock_credentials, dry_run=dry_run)
    destination_path = "directory/file.txt"
    expected_dir = mock_credentials["root_path"] / "directory"
    mock_run.side_effect = [mocker.Mock(returncode=1), mocker.Mock(returncode=0)]
    transfer_manager.ensure_remote_dir_exists(Path(destination_path))

    mock_run.assert_any_call(  # call to test if the directory exists
        [
            "ssh",
            f"{mock_credentials['user']}@{mock_credentials['host']}",
            f"test -d {expected_dir}",
        ],
        check=False,
    )
    if not dry_run:
        mock_run.assert_any_call(  # call to create the directory (since it did not exist)
            [
                "ssh",
                f"{mock_credentials['user']}@{mock_credentials['host']}",
                f"mkdir -p {expected_dir}",
            ],
            check=True,
        )
    assert mock_run.call_count == (1 if dry_run else 2)  # ensure correct number of calls


@pytest.mark.parametrize(
    "directory_exists, dry_run",
    argvalues=[(True, False), (False, False), (False, True)],
    ids=["existing_directory", "new_directory", "dry_run"],
)
def test_ensure_local_dir_exists(tmp_path, directory_exists, dry_run):
    """
    Test for :meth:`TransferManager.ensure_local_dir_exists`.

    Test Inputs
    -----------
    directory_exists : bool
        Whether the directory structure already exists.
    dry_run : bool
        Whether the test should simulate the operations (dry_run=True) or perform them (dry_run=False).

    Expected Output
    ---------------
    Creation of the directory structure locally if it does not exist, unless in dry-run mode.
    """
    transfer_manager = TransferManager(dry_run=dry_run)
    destination_path = tmp_path / "new_dir/file.txt"  # full path, converted to Path object
    expected_dir = destination_path.parent
    if directory_exists:  # pre-create the directory if it should already exist
        expected_dir.mkdir(parents=True, exist_ok=True)
    transfer_manager.ensure_local_dir_exists(destination_path)
    if dry_run:
        assert (
            not expected_dir.exists()
        ), f"Directory {expected_dir} should not have been created in dry-run mode"
    else:
        assert expected_dir.exists(), f"Directory {expected_dir} not created"


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "normal"])
def test_run_rsync(mocker, dry_run):
    """
    Test for :meth:`TransferManager._run_rsync`.

    Test Inputs
    -----------
    source : str
        Path to the file or directory to transfer.
    destination : str
        Path to the destination file or directory.
    dry_run : bool
        Whether the test should simulate the operations (dry_run=True) or perform them (dry_run=False).

    Expected Output
    ---------------
    Call the rsync command with the correct arguments.

    Implementation
    --------------
    Mock the `subprocess.run` method to check the execution of the rsync command.
    """
    mock_run = mocker.patch("subprocess.run")
    transfer_manager = TransferManager(dry_run=dry_run)
    source = "/source/path/file.txt"
    destination = "/destination/path/file.txt"
    transfer_manager._run_rsync(source, destination)
    expected_command = ["rsync", "-avz"]
    if dry_run:
        expected_command.append("--dry-run")
    expected_command.extend([source, destination])
    mock_run.assert_called_once_with(expected_command, check=True)
