#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_utils.test_path_managers_base` [module]

See Also
--------
:mod:`utils.io.path_managers.base`: Tested module.
"""
import os

import pytest

from utils.io.path_managers.base import PathManager


def test_check_dir_existing(tmp_path):
    """
    Test :meth:`PathManager.check_dir` for an existing directory.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture at the start of the test,
        and directly passed to the test function (thus, guaranteed to exist).

    Expected Output
    ---------------
    True (no error)
    """
    assert PathManager.check_dir(tmp_path) is True, "Check failed for existing directory"


def test_check_dir_non_existing():
    """
    Test :meth:`PathManager.check_dir` for a non-existing directory.

    Test Inputs
    -----------
    String without any creation of directory (no pytest fixture is used).

    Expected Output
    ---------------
    False and FileNotFoundError
    """
    non_existing_path = "non_existing_directory"
    assert PathManager.check_dir(non_existing_path) is False, "Check failed for missing directory"


def test_create_dir(tmp_path):
    """
    Test :meth:`PathManager.create_dir` for creating a new directory.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture.

    Expected Output
    ---------------
    Create the directory and return the path.
    """
    new_dir = tmp_path / "new_directory"
    created_dir = PathManager.create_dir(new_dir)
    assert new_dir.exists(), "Directory not created"
    assert created_dir == new_dir, "Returned path does not match the created directory"


def test_get_root(tmp_path):
    """
    Test :meth:`PathManager.get_root` for the default value.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture. It is exported as an environment variable
        under the name "DATA_DIR" in the current shell.

    Expected Output
    ---------------
    Value of the environment variable "DATA_DIR".
    """
    data_dir = tmp_path
    os.environ["DATA_DIR"] = str(data_dir)
    assert PathManager.get_root() == data_dir, "Root directory not retrieved"