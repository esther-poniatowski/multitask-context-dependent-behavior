#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.test_path_system.test_local` [module]

See Also
--------
:mod:`utils.path_system.base`: Tested module.
:mod:`utils.path_system.local_server`: Tested module.

Notes
-----
This test suite is focused on the methods of the :class:`LocalServer` concrete subclass, but some
methods are inherited from the :class:`ServerInterface` abstract class.
"""
import os
from pathlib import Path
import pytest

from utils.path_system.local_server import LocalServer


@pytest.mark.parametrize(
    "root_mode", argvalues=["custom", "env", "default"], ids=["custom", "env", "default"]
)
def test_get_root(root_mode, tmp_path):
    """
    Test for :meth:.get_root`.

    Test Inputs
    -----------
    root_mode : {"custom", "env", "default"}
        Mode according to which the root path is set.
    tmp_path : pathlib.Path
        If `root_set` is True, used as the root path environment variable.

    Expected Output
    ---------------
    If `root_mode` is "custom": Set to `custom_root`.
    If `root_mode` is "env": Set to `env_root`.
    If `root_mode` is "default": Set to the default root path defined in :class:`ServerInterface`,
    i.e. Path("~/mtcdb").
    """
    # Change the current working directory to avoid trivial cases
    saved_dir = os.getcwd()
    os.chdir(tmp_path)
    # Configure the ROOT environment variable
    saved_root = os.environ.get("ROOT")
    if root_mode == "custom":
        expected_root = "custom_root"
        root_arg = expected_root
    elif root_mode == "env":  # export the ROOT environment variable in the current shell session
        expected_root = "env_root"
        root_arg = None
        os.environ["ROOT"] = expected_root
    elif root_mode == "default":  # unset the ROOT environment variable
        expected_root = LocalServer.default_root
        root_arg = None
        os.environ.pop("ROOT", None)
    # Check the root path (same in both test cases)
    server = LocalServer(root_path=root_arg)
    assert server.root_path == Path(expected_root), "Root path not set"
    # Restore the previous state of the environment
    os.chdir(saved_dir)
    if saved_root is not None:
        os.environ["ROOT"] = saved_root
    else:
        os.environ.pop("ROOT", None)


@pytest.mark.parametrize("exists", argvalues=[True, False], ids=["existing", "non-existing"])
def test_is_dir(tmp_path, exists):
    """
    Test :meth:`is_dir` for an existing and a non-existing directory.

    Test Inputs
    -----------
    tmp_path : str or Path
        Path to check if it is a directory. Created by pytest fixture, guaranteed to exist.
    exists : bool
        Flag to indicate if the directory exists.

    Expected Output
    ---------------
    True if the directory exists, False otherwise.
    """
    if exists:
        path = tmp_path
    else:
        path = "non_existing_path"
    assert LocalServer().is_dir(path) is exists, f"Check failed for path: {path}"


def test_is_dir_with_file(tmp_path):
    """
    Test :meth:`is_dir` for a file path.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture where a file is created in the test.

    Expected Output
    ---------------
    False for a file.
    """
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")
    assert LocalServer().is_dir(test_file) is False, "Check failed for file path"


def test_is_file(tmp_path):
    """
    Test :meth:`is_file` to verify a file path.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture, where a file is created in the test.

    Expected Output
    ---------------
    True for a file, False otherwise.
    """
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")
    assert LocalServer().is_file(test_file) is True, "Check failed for file path"
    assert LocalServer().is_file(tmp_path) is False, "Check failed for directory path"


def test_is_file_with_dir(tmp_path):
    """
    Test :meth:`is_file` for a directory path.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture.

    Expected Output
    ---------------
    False for a directory.
    """
    assert LocalServer().is_file(tmp_path) is False, "Check failed for directory path"


def test_check_parent(tmp_path):
    """
    Test :meth:`check_parent` for the parent directory of a file.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture, which should be the parent directory.

    Expected Output
    ---------------
    True (no error)
    """
    test_file = tmp_path / "test_file.txt"
    assert LocalServer().check_parent(test_file) is True, "Check failed for parent directory"


def test_create_dir(tmp_path):
    """
    Test :meth:`create_dir` for creating a new directory.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture where the new directory should be created.

    Expected Output
    ---------------
    Directory created and correct path returned.
    """
    new_dir = tmp_path / "new_directory"
    LocalServer().create_dir(new_dir)
    assert new_dir.exists(), "Directory not created"


@pytest.mark.parametrize(
    "filename",
    argvalues=["test", "test.csv", "test.wrong"],
    ids=["no-ext", "right-ext", "wrong-ext"],
)
def test_enforce_ext(tmp_path, filename):
    """
    Test :meth:`enforce_ext` for handling file extensions.

    Test Inputs
    -----------
    filename [no-ext] : str
        Initial filename with no extension.
    filename [right-ext] : str
        Initial filename with the correct extension (.csv).
    filename [wrong-ext] : str
        Initial filename with an incorrect extension.
    expected_filepath [no-ext] : str
        Expected file path with the appropriate extension added.
    expected_filepath [right-ext] : str
        Expected file path, identical to the input path.
    expected_filepath [wrong-ext] : str
        Expected file path with the incorrect extension replaced
        by the appropriate extension.
    """
    filepath = tmp_path / filename
    expected_filepath = tmp_path / "test.csv"
    enforced_path = LocalServer().enforce_ext(filepath, ".csv")
    assert enforced_path == expected_filepath, f"Incorrect path. Expected: {expected_filepath}"


def test_display_tree(tmp_path, capsys):
    """
    Test :meth:`display_tree` to verify directory structure display.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture. Subdirectories and files are added.
    capsys :
        Fixture to capture stdout and stderr. Here, used to capture the output of the function,
        which is printed to the console.

    Expected Output
    ---------------
    Correct directory tree structure output.
    """
    (tmp_path / "subdir1").mkdir()
    (tmp_path / "subdir2").mkdir()
    (tmp_path / "subdir2" / "file1.txt").write_text("content")
    LocalServer().display_tree(tmp_path, limit=2)
    captured = capsys.readouterr()
    assert "|-- subdir1" in captured.out, "Directory tree structure not displayed correctly"
    assert "|-- subdir2" in captured.out, "Directory tree structure not displayed correctly"
    assert "|-- ..." not in captured.out, "Unexpected ellipsis in the output"
