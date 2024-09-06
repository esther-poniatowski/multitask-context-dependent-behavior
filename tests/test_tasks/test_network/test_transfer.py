#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.test_network.test_transfer` [module]

Test the file transfer functionality of the `TransferManager` class.

Implementation
--------------
Mock data for the `.env` file and the sync map in: `tests/mock_data/input_files/`.

See Also
--------
:mod:`tasks.network.transfer`: Tested module.
:mod:`socket`: Used to get the IP address of the local machine (standard library).
"""
from pathlib import Path

import pytest

from tasks.network.transfer import TransferManager
from mock_data.match_content import PATH_MOCK_DATA


def test_check_direction():
    """
    Test for :meth:`TransferManager._check_direction`.

    Test Inputs
    -----------
    source : str
        Path to the source file or directory.
    destination : str
        Path to the destination file or directory.

    Expected Output
    ---------------
    Raise a `ValueError` if the direction is not valid.
    """
    wrong_direction = "wrong"
    with pytest.raises(ValueError):
        TransferManager(direction=wrong_direction)


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
