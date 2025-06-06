#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.test_manage_remote` [module]

See Also
--------
:mod:`manage_remote`: Tested module.

Implementation
--------------
Mock data for the `.env` file and directories: `tests/test_tasks/test_network/mock_data`.
"""
# pylint: disable=redefined-outer-name

from pathlib import Path
import pytest

from utils.path_system.remote_server import RemoteServer
from mock_data.match_content import PATH_MOCK_ENV, mock_credentials


def test_load_network_config(mock_credentials):
    """
    Test for :meth:`TransferManager.load_network_config`.

    Test Inputs
    -----------
    env_path : str
        Path to a mock `.env` file containing network credentials.
    mock_credentials : dict
        See :func:`mock_credentials`.

    Expected Output
    ---------------
    RemoteServer attributes : `user`, `host`, `root_path` loaded from the `.env` file.
    """
    env_path = PATH_MOCK_ENV
    remote_manager = RemoteServer()
    remote_manager.load_network_config(env_path)
    assert remote_manager.user == mock_credentials["user"], "User not loaded"
    assert remote_manager.host == mock_credentials["host"], "Host not loaded"
    assert remote_manager.root_path == Path(mock_credentials["root_path"]), "Root path not loaded"


@pytest.mark.parametrize(
    "directory_exists", argvalues=[True, False], ids=["existing_directory", "missing_directory"]
)
def test_is_dir(directory_exists, mock_credentials, mocker):
    """
    Test for :meth:`RemoteServer.is_dir`.

    Test Inputs
    -----------
    directory_exists : bool
        Whether the directory exists on the remote server.
    directory_path : str
        Path to the directory on the remote server relative to the root path.
        Here: `/path/to/directory`.
    mock_credentials : dict
        See :func:`mock_credentials`.

    Expected Output
    ---------------
    Execution of the appropriate SSH commands to first check if the directory exists.
    Return True if the directory exists, False otherwise.

    Implementation
    --------------
    Mock the `subprocess.run` method to simulate the SSH command.
    Simulate directory existence or absence with returncode (0: exists, 1: does not exist).
    """
    remote_manager = RemoteServer(**mock_credentials)
    directory_path = "/path/to/directory"
    directory_full_path = mock_credentials["root_path"] / directory_path
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = mocker.Mock(returncode=0 if directory_exists else 1)
    result = remote_manager.is_dir(Path(directory_path))
    mock_run.assert_called_once_with(
        [
            "ssh",
            f"{mock_credentials['user']}@{mock_credentials['host']}",
            f"test -d {directory_full_path}",
        ],
        check=False,
    )
    assert result == directory_exists, f"Expected: {directory_exists} Got: {result}"


@pytest.mark.parametrize(
    "return_code, stderr",
    argvalues=[(0, ""), (1, "Error: Permission denied")],
    ids=["success", "failure"],
)
def test_create_dir(mock_credentials, mocker, return_code, stderr):
    """
    Test for :meth:`RemoteServer.create_dir` for success and failure scenarios.

    Arguments
    ---------
    return_code : int
        Return code simulated by `subprocess.run` (0 for success, 1 for failure).
    stderr : str
        Error message outputted in case of a failure.

    Test Inputs
    -----------
    directory_path : str
        Path to the directory on the remote server relative to the root path.
        Here: `/path/to/directory`.
    mock_credentials : dict
        See :func:`mock_credentials`.

    Expected Output
    ---------------
    Execution of the appropriate SSH commands to create the directory.

    Implementation
    --------------
    Mock the `subprocess.run` method to simulate the SSH command.
    """
    remote_manager = RemoteServer(**mock_credentials)
    directory_path = "/path/to/directory"
    directory_full_path = Path(directory_path)
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = mocker.Mock(returncode=return_code, stderr=stderr)
    remote_manager.create_dir(Path(directory_path))  # expected to build the full path
    mock_run.assert_called_once_with(
        [
            "ssh",
            f"{mock_credentials['user']}@{mock_credentials['host']}",
            f"mkdir -p {directory_full_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
