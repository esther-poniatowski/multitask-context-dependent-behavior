#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.test_path_system.test_explorer` [module]

See Also
--------
:mod:`utils.path_system.explorer`: Tested module.
"""
import os
from pathlib import Path

import pytest

from utils.path_system.explorer import (
    check_path,
    is_file,
    check_parent,
    create_dir,
    enforce_ext,
    display_tree,
)


def test_check_path_existing(tmp_path):
    """
    Test :func:`check_path` for an existing path.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture, guaranteed to exist.

    Expected Output
    ---------------
    True (no error)
    """
    assert check_path(tmp_path) is True, "Check failed for existing path"


def test_check_path_non_existing():
    """
    Test :func:`check_path` for a non-existing path.

    Test Inputs
    -----------
    String without any creation of directory (no pytest fixture is used).

    Expected Output
    ---------------
    False and FileNotFoundError (if raise_error=True)
    """
    non_existing_path = "non_existing_path"
    assert check_path(non_existing_path) is False, "Check failed for missing path"
    with pytest.raises(FileNotFoundError):
        check_path(non_existing_path, raise_error=True)


def test_is_file(tmp_path):
    """
    Test :func:`is_file` to verify a file path.

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
    assert is_file(test_file) is True, "Check failed for file path"
    assert is_file(tmp_path) is False, "Check failed for directory path"


def test_check_parent(tmp_path):
    """
    Test :func:`check_parent` for the parent directory of a file.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture, which should be the parent directory.

    Expected Output
    ---------------
    True (no error)
    """
    test_file = tmp_path / "test_file.txt"
    assert check_parent(test_file) is True, "Check failed for parent directory"


def test_create_dir(tmp_path):
    """
    Test :func:`create_dir` for creating a new directory.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture where the new directory should be created.

    Expected Output
    ---------------
    Directory created and correct path returned.
    """
    new_dir = tmp_path / "new_directory"
    created_dir = create_dir(new_dir)
    assert new_dir.exists(), "Directory not created"
    assert created_dir == new_dir, "Returned path does not match the created directory"


@pytest.mark.parametrize(
    "filename",
    argvalues=["test", "test.csv", "test.wrong"],
    ids=["no-ext", "right-ext", "wrong-ext"],
)
def test_enforce_ext(tmp_path, filename):
    """
    Test :func:`enforce_ext` for handling file extensions.

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
    enforced_path = enforce_ext(filepath, ".csv")
    assert enforced_path == expected_filepath, f"Incorrect path. Expected: {expected_filepath}"


def test_display_tree(tmp_path, capsys):
    """
    Test :func:`display_tree` to verify directory structure display.

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
    display_tree(tmp_path, limit=2)
    captured = capsys.readouterr()
    assert "|-- subdir1" in captured.out, "Directory tree structure not displayed correctly"
    assert "|-- subdir2" in captured.out, "Directory tree structure not displayed correctly"
    assert "|-- ..." not in captured.out, "Unexpected ellipsis in the output"
