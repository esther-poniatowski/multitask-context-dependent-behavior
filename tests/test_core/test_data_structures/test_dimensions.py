#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`test_core.test_data_structures.test_dimensions` [module]

See Also
--------
`core.data_structures.dimensions`: Tested module.
"""

import pytest

from core.data_structures.dimensions import DimName, Dimensions


# --- Test DimName class ---------------------------------------------------------------------------


def test_valid_dim_name():
    """
    Test the `DimName` class with a valid dimension name.
    """
    DimName("time")


def test_invalid_dim_name():
    """
    Test the `DimName` class with an invalid dimension name.
    """
    with pytest.raises(ValueError):
        DimName("invalid_dim")


def test_dim_alias():
    """
    Test the property `DimName.alias` to get the alias of a dimension name.
    """
    dim = DimName("time")
    assert dim.alias == DimName._ALIASES["time"]  # pylint: disable=protected-access


# --- Test Dimensions class ------------------------------------------------------------------------


def test_dimensions_init():
    """
    Test the instantiation of the `Dimensions` class:

    - By passing dimension names as positional arguments.
    - By unpacking a tuple of dimension names.
    - By passing a tuple of dimension names.
    """
    names = ("time", "units")
    # Pass dimension names as positional arguments
    dims = Dimensions("time", "units")
    assert dims == names
    # Unpack a tuple of dimension names
    dims = Dimensions(*names)
    assert dims == names
    # Pass a tuple of dimension names
    dims = Dimensions(names)
    assert dims == names


def test_get_dim():
    """
    Test the `get_dim` method of the `Dimensions` class.

    Test Inputs
    -----------
    axis : int
        Index of the dimension to retrieve.

    Expected Output
    ---------------
    If the axis if inferior to the number of dimensions, the name of the dimension is returned.
    Otherwise, an IndexError is raised.
    """
    dims = Dimensions("time", "units")
    axis = 0
    assert dims.get_dim(axis) == dims[axis]
    axis = 100
    with pytest.raises(IndexError):
        dims.get_dim(axis)


def test_get_axis():
    """
    Test the `get_axis` method delegated to the `Dimensions` class.

    Test Inputs
    -----------
    dim : int
        Name the dimension to retrieve.

    Expected Output
    ---------------
    If the dimension name exists in the object, the index of the dimension is returned.
    Otherwise, a ValueError is raised.
    """
    dims = Dimensions("time", "units")
    assert dims.get_axis("time") == 0
    with pytest.raises(ValueError):
        dims.get_axis("invalid")
