#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_datasets.test_utils` [module]

Tests for the module :mod:`mtcdb.datasets.utils`.
"""

import numpy as np
from numpy.typing import NDArray
import os
import pytest 

from mtcdb.datasets.utils import DataHandler


@pytest.fixture
def setup_test_data(tmp_path) -> tuple[NDArray, str]:
    """
    Create test data and a corresponding temporary .npy file path.

    Returns
    -------
    test_data: NDArray
        Numpy array to be saved and/or loaded.
    test_file: str
        Temporary path.
    
    Note
    ----
    This fixture uses itself pytest's fixture :obj:`tmp_path`.
    The test file is automatically deleted by pytest after the test is run.
    """
    test_data: NDArray = np.array([1, 2, 3])
    test_file = tmp_path / "test.npy"
    return test_data, str(test_file)


def test_load_data_success(setup_test_data):
    """
    Test for the method :func:`load` in the mixin class :class:`DataHandler`.
    
    Define a *valid* subclass using the mixin class to test.
    Instantiate a subclass object and loads test data.
    
    Test Inputs
    -----------
    test_data: NDArray
        Numpy array to be loaded.
    TestNpyDataHandler: 
        Valid subclass.
        It possesses all the attributes required by :class:`DataHandler`.
        Its attribute ``data_path`` corresponds to the file created in the test.
    """
    class TestNpyDataHandler(DataHandler):
        def __init__(self):
            self.data_path = str(test_file)
            self.data_loader = np.load
            self.empty_shape = (0,)
    
    test_data, test_file = setup_test_data
    np.save(test_file, test_data)
    test_instance = TestNpyDataHandler()
    data = test_instance.load()
    assert np.array_equal(data, test_data), "Not the expected data."


def test_load_data_file_not_found():
    """
    Test for the method :func:`load` in the mixin class :class:`DataHandler`.

    Define an *invalid* subclass using the mixin class to test,
    which points to a non-existent file.

    Test Inputs
    -----------
    TestNpyDataHandler: 
        Invalid subclass.
        It possesses all the attributes required by :class:`DataHandler`.
        Its attribute ``data_path`` does not corresponds to any existing file.
    
    Expected Outputs
    ----------------
    data: NDArray
        Empty data of shape ``empty_shape``.
    """
    class TestNpyDataHandler(DataHandler):
        def __init__(self):
            self.data_path = "/invalid/test.npy"
            self.data_loader = np.load
            self.empty_shape = (0,0)
    
    test_instance = TestNpyDataHandler()
    data = test_instance.load()
    assert data.shape == test_instance.empty_shape, "Invalid shape."
    assert data.size == 0, "Not empty."


def test_load_data_missing_attribute():
    """
    Test for the method :func:`load` in the mixin class :class:`DataHandler`.

    Define an *invalid* subclass using the mixin class to test,
    which lacks one of the required attributes.

    Test Inputs
    -----------
    TestNpyDataHandler: 
        Invalid subclass.
        It lacks the attribute ``data_path``.
    
    Expected Outputs
    ----------------
    NotImplementedError
        The instanciation of the subclass itself should raise an error.
    
    Note 
    ----
    No need to generate test data, because in the implementation of 
    :class:`DataHandler` the presence of attributes if checked even
    before trying to recover data.
    """
    class TestNpyDataHandler(DataHandler):
        def __init__(self):
            self.data_loader = np.load
            self.empty_shape = (0,0)
    
    test_instance = TestNpyDataHandler()
    with pytest.raises(NotImplementedError):
        test_instance.load()

    
def test_save_data_success(setup_test_data):
    """
    Test for the method the method :func:`save` in the mixin class :class:`DataHandler`.
     
    Define a *valid* subclass to save data to an existing directory.
    
    Test Inputs
    -----------
    test_data: NDArray
        Numpy array to be saved.
    TestNpyDataHandler:
        Valid subclass with required attributes for saving data.
    
    Expected Outputs
    ----------------
    test_file: Path
        The file path where the data was saved,
        recovered using :func:`np.load`.
    """    
    class TestNpyDataHandler(DataHandler):
        def __init__(self):
            self.data_path = str(test_file)
            self.data_saver = np.save

    test_data, test_file = setup_test_data
    test_instance = TestNpyDataHandler()
    test_instance.save(test_data)
    assert np.array_equal(np.load(test_file), test_data), "Saved data does not match expected data."


def test_save_data_directory_not_found(setup_test_data):
    """
    Test for the method the method :func:`save` in the mixin class :class:`DataHandler`.

    Define a *invalid* subclass to save data to a non-existent directory.

    Test Inputs
    -----------
    test_data: ndarray
        Numpy array to attempt to save in a non-existent directory.
    TestNpyDataHandler:
        Subclass with a non-existent directory path.
    
    Expected Outputs
    ----------------
    FileNotFoundError
        An error should be raised indicating the directory does not exist.
    """
    class TestNpyDataHandler(DataHandler):
        def __init__(self):
            self.data_path = "/invalid/test.npy"
            self.data_saver = np.save
    
    test_data, _ = setup_test_data
    test_instance = TestNpyDataHandler()
    with pytest.raises(FileNotFoundError):
        test_instance.save(test_data)

