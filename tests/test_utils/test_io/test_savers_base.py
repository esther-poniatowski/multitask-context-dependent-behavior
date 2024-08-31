#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_io_data.test_savers_base` [module]

Notes
-----
Although those tests aim to cover the method defined in the abstract base class,
sometimes the SaverCSV subclass is used for concrete implementation.
It is chosen for simplicity and because it offers more options to test.
Those tests do not check for the concents of the data saved,
but rather for the correct handling of the file paths and formats.

See Also
--------
:mod:`utils.io_data.savers.base`: Tested module.
:mod:`utils.io_data.savers.impl`: Concrete implementations.
"""

import pytest

from test_utils.test_io.mock_data import (  # dummy data
    data_list,
    data_dict,
    data_array,
    data_df,
)
from utils.io_data.savers.impl import SaverCSV


@pytest.mark.parametrize(
    "data", argvalues=[data_list, data_array, data_df], ids=["list", "ndarray", "dataframe"]
)
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
    saver._check_data()


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
        saver._check_data()
