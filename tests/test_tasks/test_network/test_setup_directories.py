#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_tasks.test_network.test_setup_directories` [module]

Test the directory setup functionality of the `DirectoryOrganizer` class.

Implementation
--------------
Mock data for YAML directory structure and `.env` files.

See Also
--------
:mod:`tasks.network.setup_directories`: Tested module.
:mod:`dotenv`: Used to load environment variables (local or remote).
"""
from pathlib import Path

import pytest

from mock_data.match_content import PATH_MOCK_YAML, mock_structure, expected_paths
from tasks.network.setup_directories import DirectoryOrganizer
from utils.path_system.remote_server import RemoteServer


@pytest.mark.parametrize("remote", argvalues=[False, True], ids=["local", "remote"])
def test_init_directory_organizer(remote):
    """
    Test for :meth:`DirectoryOrganizer.__init__`.

    Expected Output
    ---------------
    DirectoryOrganizer attributes:
    - `dry_run` as a boolean.
    - `remote` as a boolean.
    - `root_path` as a Path object.
    - `directory_structure` as None.
    """
    organizer = DirectoryOrganizer(dry_run=True, remote=remote)
    assert organizer.dry_run is True
    assert isinstance(organizer.server.root_path, Path)
    assert isinstance(organizer.directory_structure, dict)


@pytest.mark.parametrize(
    "root_path_provided, remote",
    argvalues=[(True, True), (False, True), (False, False)],
    ids=["root_path_provided", "remote_default", "local_default"],
)
def test_root_path(root_path_provided, remote, tmp_path):
    """
    Test for correct assignment of the attribute :attr:`root_path` depending on the case.

    Test Inputs
    -----------
    root_path_provided : bool
        Whether the `root_path` argument is provided.
    remote : bool
        Value of the `remote` argument.
    tmp_path : pathlib.Path
        Temporary directory path, used as the argument `root_path` in the test.

    Expected Output
    ---------------
    Set the attribute :attr:`root_path` depending on the case:

    - If `root_path` is provided: Use it as the value.
    - Otherwise, if `remote` is True: Use the default value defined as the class attribute
      :attr:`default_root` in the base class :class:`ServerInterface`.
    - Otherwise, if `remote` is False: Use the method :meth:`get_root` in the class
      :class:`LocalServer`.
    """
    if root_path_provided:
        root_arg = "test_root"
        expected_attr = Path(root_arg)
        organizer = DirectoryOrganizer(root_path=root_arg, remote=remote)
        assert organizer.server.root_path == expected_attr
    else:
        organizer = DirectoryOrganizer(remote=remote)
        if remote:
            assert organizer.server.root_path == RemoteServer.default_root
        else:
            assert organizer.server.root_path == organizer.server.get_root()


def test_load_directory_structure(mock_structure):
    """
    Test for :meth:`DirectoryOrganizer.load_directory_structure`.

    Test Inputs
    -----------
    yaml_path : str
        Path to a mock YAML file containing the directory structure.
    mock_structure : dict
        See :func:`mock_structure`.

    Expected Output
    ---------------
    DirectoryOrganizer attribute : `directory_structure` should be a dictionary.
    """
    yml_path = PATH_MOCK_YAML
    organizer = DirectoryOrganizer(dry_run=False, remote=False)
    organizer.load_directory_structure(yml_path)
    assert isinstance(
        organizer.directory_structure, dict
    ), "Directory structure not loaded as a dictionary"
    assert organizer.directory_structure == mock_structure


@pytest.mark.parametrize("dry_run", argvalues=[True, False], ids=["dry_run", "normal"])
def test_create_directories(mock_structure, tmp_path, expected_paths, dry_run):
    """
    Test for :meth:`DirectoryOrganizer.create_directories`.

    Test Inputs
    -----------
    mock_structure : dict
        Directory structure to create. See :func:`mock_structure`.
    tmp_path : pathlib.Path
        Temporary directory path used as the root directory.
    expected_paths : list
        List of expected paths based on the directory structure. See :func:`expected_paths`.
    dry_run : bool
        Whether operations should be simulated or performed.

    Expected Output
    ---------------
    Creation of the directory structure locally, unless in dry-run mode.
    """
    organizer = DirectoryOrganizer(
        root_path=tmp_path, directory_structure=mock_structure, dry_run=dry_run, remote=False
    )
    organizer.create_directories()  # no arguments, to use the attributes as defaults
    if not dry_run:  # check directories' creation
        for path in expected_paths:
            assert path.exists(), f"Directory {path} not created"
    else:  # check directories' simulation
        for path in expected_paths:
            assert (
                not path.exists()
            ), f"Directory {path} should not have been created in dry-run mode"
