#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.path_system.explorer` [module]

Functions
---------
:func:`check_path`
:func:`is_file`
:func:`check_parent`
:func:`create_dir`
:func:`display_tree`
"""
from pathlib import Path
from typing import Union

from utils.io.formats import FileExt


def check_path(path: Union[str, Path], raise_error: bool = False) -> bool:
    """
    Check the existence of a path in the file system.

    Parameters
    ----------
    path: str or Path
    raise_error: bool, default=False
        Whether to raise an error if the path does not exist.

    Returns
    -------
    bool
        True if the path exists, False otherwise.

    Raises
    ------
    FileNotFoundError
        If the path does not exist and `raise_error` is True.

    See Also
    --------
    :func:`pathlib.exists`
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        if raise_error:
            raise FileNotFoundError(f"Inexistent path: {path}")
        return False
    return True


def is_file(path: Union[str, Path]) -> bool:
    """
    Check if a path corresponds to a file.

    Parameters
    ----------
    path: str or Path

    Returns
    -------
    bool

    See Also
    --------
    :func:`pathlib.is_file`
    """
    if isinstance(path, str):
        path = Path(path)
    return path.is_file()


def check_parent(path: Union[str, Path]) -> bool:
    """
    Check the existence of the parent directory of a file or other directory.

    Parameters
    ----------
    path: str or Path

    Returns
    -------
    bool

    See Also
    --------
    :func:`pathlib.parent`
    """
    if isinstance(path, str):
        path = Path(path)
    return check_path(path.parent)


def create_dir(path: Union[str, Path]) -> Path:
    """
    Create a directory at a given path if it does not exist.

    Parameters
    ----------
    path: str or Path

    See Also
    --------
    :func:`pathlib.mkdir`: Create a directory.
        Parameter `parents` : Create parent directories if needed.
        Parameter `exist_ok` : If the directory already exists, nothing is done.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path}")
    else:
        print(f"Pre-existing directory: {path}")
    return path


def enforce_ext(path: Union[str, Path], ext: Union[str, FileExt]) -> Path:
    """
    Enforce a specific file extension on a path.

    If the file extension is missing or incorrect, it is added or corrected.

    Parameters
    ----------
    path: str or Path
    ext: str or FileExt
        File extension to enforce.

    Returns
    -------
    Path
        Path with the correct file extension.

    Raises
    ------
    ValueError
        If the extension does not start with a period.

    See Also
    --------
    :meth:`pathlib.Path.with_suffix`
        If there is already an extension, it is replaced.
        If there is no extension, it is added.
    """
    if isinstance(path, str):
        path = Path(path)
    if isinstance(ext, FileExt):
        ext = ext.value  # convert to string
    if not ext.startswith("."):
        raise ValueError(f"Invalid extension: {ext}")
    return path.with_suffix(ext)


def display_tree(path: Union[str, Path], level: int = 0, limit: int = 5) -> None:
    """
    Display the tree structure of a directory.

    Parameters
    ----------
    path : str or Path
    level : int, optional
        Current level in the directory tree, used for indentation.
    limit : int, optional
        Maximum number of items to display per directory.

    Implementation
    --------------

    - Display the name of each file or subdirectory in the currently traversed directory, with an
      indentation level depending on its depth in the hierarchy.
    - If a directory contains more items than the specified limit, show an ellipsis (`...`).
    - Call the method recursively on sub-directories until reaching the end of the directory
      hierarchy (i.e when encountering a directory that contains no subdirectories or files).
    """
    if isinstance(path, str):
        path = Path(path)
    check_path(path, raise_error=False)
    items = list(path.iterdir())
    display_items = items[:limit]
    for item in display_items:
        print("    " * level + "|-- " + item.name)
        if item.is_dir():
            display_tree(item, level + 1, limit)
    if len(items) > limit:
        print("    " * level + "|-- ...")
