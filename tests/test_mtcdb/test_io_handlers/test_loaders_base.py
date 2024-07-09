#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_io_handlers.test_loaders_base` [module]

Notes
-----
Although those tests aim to cover the method defined in the abstract base class, 
sometimes the LoaderCSV subclass is used for concrete implementation.
It is chosen for simplicity and because it offers more options to test.
Those tests do not check for the concents of the data loaded,
but rather for the correct handling of the file paths and formats.

See Also
--------
:mod:`mtcdb.io_handlers.loaders.base`: Tested module.
:mod:`mtcdb.io_handlers.loaders.impl`: Concrete implementations.
"""

import pytest

from mtcdb.io_handlers.loaders.base import Loader
from mtcdb.io_handlers.loaders.impl import LoaderCSV


def test_loader_check_file_invalid():
    """
    Test :meth:`Loader._check_file` for a non-existent file.

    Test Inputs
    -----------
    filepath : str
        Path to a non-existent file.

    Expected Exception
    ------------------
    FileNotFoundError
    """
    filepath = "invalid_path"
    loader = Loader(filepath, tpe='list')
    with pytest.raises(FileNotFoundError):
        loader._check_file()

def test_loader_check_file_valid(tmp_path):
    """
    Test :meth:`Loader._check_file` for an existent file.

    Test Inputs
    -----------
    filepath : str
        Path to an existent file (temporary directory).

    Expected Output
    ---------------
    No exception raised.
    """
    filepath = tmp_path
    loader = Loader(filepath, tpe='list')
    loader._check_file()


def test_loader_check_type_invalid():
    """
    Test :meth:`LoaderCSV._check_type` for an invalid data type.

    Test Inputs
    -----------
    tpe : str
        Invalid data type for the CSV loader : 'dict'.

    Expected Exception
    ------------------
    ValueError
    """
    loader = LoaderCSV("test.csv", tpe='dict')
    with pytest.raises(TypeError):
        loader._check_type()


def test_loader_check_type_valid():
    """
    Test :meth:`LoaderCSV._check_type` for a valid data type.

    Test Inputs
    -----------
    tpe : str
        Valid data type for the CSV loader : 'list'.

    Expected Output
    ---------------
    No exception raised.
    """
    loader = LoaderCSV("test.csv", tpe='list')
    loader._check_type()
