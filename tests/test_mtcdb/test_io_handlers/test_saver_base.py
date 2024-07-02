#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_io_handlers.test_saver_base` [module]

Notes
-----
Although those tests aim to cover the method defined in the abstract base class, 
sometimes the SaverCSV subclass is used for concrete implementation.
It is chosen for simplicity and because it offers more options to test.
Those tests do not check for the concents of the data saved,
but rather for the correct handling of the file paths and formats.

See Also
--------
:mod:`mtcdb.io_handlers.saver_base`: Tested module.
:mod:`mtcdb.io_handlers.savers`: Concrete implementations.
`tmp_path`: Fixture in pytest for temporary directories.
"""

import pytest

from test_mtcdb.test_io_handlers.test_data import ( # dummy data
    data_list,
    data_dict,
    data_array,
    data_df,
)
from mtcdb.io_handlers.saver_base import Saver
from mtcdb.io_handlers.savers import SaverCSV


def test_saver_check_dir_invalid():
    """
    Test :meth:`Saver._check_dir` for a non-existent directory.

    Test Inputs
    -----------
    invalid_path: str
        Path to a non-existent directory.
    
    Expected Exception
    ------------------
    FileNotFoundError
    """
    invalid_path = "/invalid_directory/test.csv"
    with pytest.raises(FileNotFoundError, match="Inexistent directory"):
        saver = Saver(invalid_path, data_list)
        saver._check_dir() # pylint: disable=protected-access


def test_saver_check_dir_valid(tmp_path):
    """
    Test :meth:`Saver._check_dir` for an existent directory.

    Test Inputs
    -----------
    valid_path: str
        Path to an existent directory.
    
    Expected Output
    ---------------
    No exception raised.
    """
    valid_path = tmp_path / "test.csv"
    saver = Saver(valid_path, data_list)
    saver._check_dir() # pylint: disable=protected-access


@pytest.mark.parametrize("filename",
                         argvalues=["test", "test.csv", "test.wrong"],
                         ids=["no-ext", "right-ext", "wrong-ext"])
def test_saver_check_ext(tmp_path, filename):
    """
    Test :meth:`SaverCSV._check_ext` for handling file extensions.

    Test Inputs
    -----------
    filename [no-ext] : str
        Initial filename with no extension.
    filename [right-ext] : str
        Initial filename with right extension (.csv).
    filename [wrong-ext] : str
        Initial filename with wrong extension.
    expected_filepath [no-ext] : str
        Expected file path with the appropriate extension added.
    expected_filepath [right-ext] : str
        Expected file path, identical to the input path.
    expected_filepath [wrong-ext] : str
        Expected file path with the wrong extension removed 
        and the appropriate extension added.
    """
    filepath = tmp_path / filename
    expected_filepath = tmp_path / "test.csv"
    saver = SaverCSV(filepath, data_list)
    saver._check_ext() # pylint: disable=protected-access
    assert saver.path == expected_filepath, f"Incorrect path. Expected: {expected_filepath}"


@pytest.mark.parametrize("data",
                         argvalues=[data_list, data_array, data_df],
                         ids=["list", "ndarray", "dataframe"])
def test_saver_check_data_valid(data):
    """
    Test :meth:`SaverCSV._check_data` for valid data for CSV.

    Test Inputs
    -----------
    data : list, np.ndarray, pd.DataFrame
        Dummy data to test, in one of the three formats supported by SaverCSV.
    
    Expected Output
    ---------------
    No exception raised.
    """
    saver = SaverCSV("test", data)
    saver._check_data() # pylint: disable=protected-access

def test_saver_check_data_invalid():
    """
    Test :meth:`SaverCSV._check_data` for invalid data for CSV.

    Test Inputs
    -----------
    data_dict : dict
        Dummy data to test in an unsupported format for SaverCSV (dict).
    
    Expected Exception
    ------------------
    TypeError
    """
    saver = SaverCSV("test", data_dict)
    with pytest.raises(TypeError):
        saver._check_data() # pylint: disable=protected-access
