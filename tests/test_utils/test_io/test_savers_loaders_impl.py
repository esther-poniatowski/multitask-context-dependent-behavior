#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.test_io.test_savers_loaders_impl` [module]

Notes
-----
Saver and Loader subclasses are tested together to ensure their consistent interaction. Contrary to
the tests in :mod:`test_core.test_io_data.test_saver_base` and :mod:`test_core.io_data.test_loader_base`,
here the focus is on the specific *content* of the data which is saved and loaded, rather than on
general checks carried out in the base classes.

Implementation
--------------
Although the saver can automatically set appropriate extensions, here it is necessary to append
paths with appropriate extensions in order to recover the data manually without redefining the path.

See Also
--------
:mod:`utils.io_data.savers.impl_savers`: Tested module.
:mod:`utils.io_data.loaders.impl_loaders`: Tested module.
"""
from pathlib import Path

import numpy as np
import pytest

from mock_data.python_objects import (
    data_list,
    data_dict,
    data_array,
    data_df,
    data_array_float,
    data_array_int,
    data_array_str,
    data_obj,
    expected_from_list,
)
from mock_data.match_content import PATH_MOCK_DATA, data_list_yml, data_dict_yml
from utils.io_data.loaders.impl_loaders import (
    LoaderCSV,
    LoaderNPY,
    LoaderPKL,
    LoaderDILL,
    LoaderYAML,
)
from utils.io_data.savers.impl_savers import SaverCSV, SaverNPY, SaverPKL, SaverDILL


@pytest.mark.parametrize(
    "data, expected, tpe",
    argvalues=[
        (data_list, expected_from_list, "list"),
        (data_array_float, data_array_float, "ndarray_float"),
        (data_array_int, data_array_int, "ndarray_int"),
        (data_array_str, data_array_str, "ndarray_str"),
        (data_df, data_df, "dataframe"),
    ],
    ids=["list", "numpy_float", "numpy_int", "numpy_str", "dataframe"],
)
def test_saver_loader_csv(tmp_path, data, expected, tpe):
    """
    Test for :class:`saver_module.SaverCSV` and :class:`loader_module.LoaderCSV`.

    Test Inputs
    -----------
    data [list]
        List of lists with headers and integer values.
    data [numpy_float, numpy_int, numpy_str]
        NumPy array with float, integer or string values (no header).
    data [dataframe]
        Pandas DataFrame with headers and integer values.

    Expected Output
    ---------------
    expected [list]
        List of lists, with *string* values.
    expected [numpy_float, numpy_int, numpy_str]
        NumPy array identical to the input data.
    expected [dataframe]
        Pandas DataFrame identical to the input data.

    Implementation
    --------------
    For checking equality between the loaded and expected data,
    the appropriate method should be used according to the data type.
    """
    # Save a CSV file with the desired data using the Saver class
    filepath = tmp_path / "test.csv"
    saver = SaverCSV(filepath, data)
    saver.save()
    # Load the data
    loader = LoaderCSV(filepath, tpe)
    content = loader.load()
    # Compare the loaded data with the expected data
    if tpe == "list":
        assert content == expected, "Content mismatch"
    elif "ndarray" in tpe:
        assert np.array_equal(content, expected), "Content mismatch"
    elif tpe == "dataframe":
        assert content.equals(expected), "Content mismatch"


def test_saver_loader_npy(tmp_path):
    """
    Test for :class:`saver_module.SaverNPY` and :class:`loader_module.LoaderNPY`.

    Test Inputs
    -----------
    data : numpy.ndarray
        NumPy array with float values.

    Expected Output
    ---------------
    data : numpy.ndarray
        NumPy array identical to the input data.
    """
    # Save a NPY file with the desired data using the Saver class
    filepath = tmp_path / "test.npy"
    saver = SaverNPY(filepath, data_array)
    saver.save()
    # Load the data
    loader = LoaderNPY(filepath)
    content = loader.load()
    # Compare the loaded data with the expected data
    assert np.array_equal(content, data_array), "Content mismatch"


@pytest.mark.parametrize(
    "data, is_custom_class",
    argvalues=[(data_dict, False), (data_obj, True)],
    ids=["dict", "custom_class"],
)
def test_saver_loader_pkl(tmp_path, data, is_custom_class):
    """
    Test for :class:`saver_module.SaverPKL` and :class:`loader_module.LoaderPKL`.

    Test Inputs
    -----------
    data [dict, custom_class]: dict or MyClass
        Sample data to be saved.
    is_custom_class: bool
        Whether the data is an instance of a custom class.

    Expected Output
    ---------------
    expected [dict, custom_class]: dict or MyClass
        Content of the pickle file identical to the input data.

    Implementation
    --------------
    If the data is a dictionary, compare each item with :func:`np.array_equal`.
    In this case, it is not possible to use simple equality ``content == expected``,
    because the values are NumPy arrays. Doing so would raise an error:
    ``ValueError: The truth value of an array with more than one element is ambiguous.``
    If the data is a custom class, use the :meth:`__eq__` method for comparison.
    """
    # Save a Pickle file with the desired data using the Saver class
    filepath = tmp_path / "test.pkl"
    saver = SaverPKL(filepath, data)
    saver.save()
    # Load the data
    loader = LoaderPKL(filepath)
    content = loader.load()
    # Compare the loaded data with the expected data
    if is_custom_class:
        assert content == data, "Content mismatch"
    else:
        for key, value in data.items():
            assert np.array_equal(content[key], value), f"Content mismatch for {key}"


def test_saver_loader_dill(tmp_path):
    """
    Test for :class:`saver_module.SaverDILL` and :class:`loader_module.LoaderDILL`.

    Test Inputs
    -----------
    data [dict, custom_class]: dict or MyClass
        Sample data to be saved.

    Expected Output
    ---------------
    expected [dict, custom_class]: dict or MyClass
        Content of the Dill file identical to the input data.

    Implementation
    --------------
    Similar to the Pickle test, the data is saved and loaded back using Dill.
    If the data is a dictionary, each item should be compared using :func:`np.array_equal`.
    If the data is a custom class, use the :meth:`__eq__` method for comparison.
    """
    filepath = tmp_path / "test.dill"
    saver = SaverDILL(filepath, data_dict)
    saver.save()
    loader = LoaderDILL(filepath)
    content = loader.load()
    for key, value in data_dict.items():
        assert np.array_equal(content[key], value), f"Content mismatch for {key}"


@pytest.mark.parametrize(
    "tpe, expected, filename",
    argvalues=[
        ("list", data_list_yml, "data_list.yml"),
        ("dict", data_dict_yml, "data_dict.yml"),
    ],
    ids=["list", "dict"],
)
def test_loader_yaml(tpe, expected, filename):
    """
    Test for :class:`loader_module.LoaderYAML`.

    Test Inputs
    -----------
    data [dict, list]: dict or list
        Sample data to be saved in YAML format.

    Expected Output
    ---------------
    expected [dict, list]: dict or list
        Content of the YAML file identical to the input data.

    Implementation
    --------------
    Retrieve data from the mock data directory whose path is relative to the current python file.
    """
    filepath = PATH_MOCK_DATA / filename
    loader = LoaderYAML(filepath, tpe)
    content = loader.load()
    assert content == expected, "Content mismatch"
