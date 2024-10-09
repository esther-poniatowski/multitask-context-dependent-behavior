#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`test_core.test_data_structures.core_data` [module]

See Also
--------
`core.data_structures.core_data`: Tested module.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportRedeclaration=false
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=redefined-outer-name


from types import MappingProxyType

import numpy as np
import pytest

from core.data_structures.core_data import DimName, CoreData


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
    assert dim.alias == DimName._ALIASES["time"]


# --- Test CoreData class --------------------------------------------------------------------------


@pytest.mark.parametrize("dims", argvalues=[None, ("time", "time")], ids=["default", "with_dims"])
def test_numpy_inheritance(dims):
    """
    Test the instantiation of a core data object which should behave as a numpy array.

    Test Inputs
    -----------
    dims : Tuple[str]
        Dimensions of the data. Warning: Valid names are defined in the `DimName` class.
    values : np.ndarray
        Values to store in the core data object. The number of dimensions should match the that in
        the `dims` argument (except when `dims` is `None`).

    Expected Output
    ---------------
    The object is created with the values passed as an argument.
    The equality is preserved with the values (as a numpy array).

    """
    # Create an array with as many dimensions as the length of `dims`
    if dims is None:
        shape = (5, 10)
    else:
        shape = tuple(np.random.randint(5, 10) for _ in dims)
    values = np.zeros(shape)
    data = CoreData(values, dims=dims)
    # Check behavior as a numpy array
    assert np.array_equal(data, values)
    assert data.shape == values.shape
    assert data.ndim == values.ndim


@pytest.mark.parametrize(
    "dims", argvalues=[("time",), ("invalid", "time")], ids=["missing_dim", "invalid_name"]
)
def test_invalid_instantiation(dims):
    """
    Test the instantiation of a core data object with a mismatch in the number of dimensions.

    Test Inputs
    -----------
    values : np.ndarray
        2D array.
    dims : Tuple[str]
        Invalid dimension. Case 1: missing dimension. Case 2: invalid dimension name.

    Expected Output
    ---------------
    Case 1: ValueError is raised by the ``__new__`` method.
    Case 2: ValueError is raised by the ``DimName`` class within the ``__new__`` method.
    """
    values = np.zeros((5, 10))
    with pytest.raises(ValueError):
        CoreData(values, dims=dims)


@pytest.mark.parametrize("axis", argvalues=[0, 100], ids=["valid_idx", "out_of_bounds"])
def test_get_dim(axis):
    """
    Test the `get_dim` method to retrieve the name of a specific dimension by index.

    Test Inputs
    -----------
    axis : int
        Index of the dimension to retrieve.

    Expected Output
    ---------------
    If the axis if inferior to the number of dimensions, the name of the dimension is returned.
    Otherwise, an IdexError is raised.
    """
    data = CoreData(np.zeros((5, 10)), dims=("time", "units"))
    if axis < data.ndim:
        assert data.get_dim(axis) == data.dims[axis]
    else:
        with pytest.raises(IndexError):
            data.get_dim(axis)


@pytest.mark.parametrize("dim", argvalues=["time", "invalid"], ids=["existent", "inexistent"])
def test_get_axis(dim):
    """
    Test the `get_axis` method to retrieve the index of a specific dimension by name.

    Test Inputs
    -----------
    dim : int
        Name the dimension to retrieve.

    Expected Output
    ---------------
    If the dimension name exists in the object, the index of the dimension is returned.
    Otherwise, a ValueError is raised.
    """
    data = CoreData(np.zeros((5, 10)), dims=("time", "units"))
    if dim in data.dims:
        assert data.get_axis(dim) == data.dims.index(dim)
    else:
        with pytest.raises(ValueError):
            data.get_axis(dim)


def test_get_size():
    """
    Test the `get_size` method to retrieve the size of the data along a specific dimension.
    """
    shape = (5, 10)
    dims = ("time", "units")
    data = CoreData(np.zeros(shape), dims=dims)
    for name, size in zip(dims, shape):
        assert data.get_size(name) == size


def test_array_finalize_preserve_dims():
    """
    Test the `__array_finalize__` method for the transmission of the `dims` attributes in
    operations which *preserve* the number of dimensions.

    Operations tested:
    - Slicing to extract a subset of the data
    """
    data = CoreData(np.zeros((10, 10)), dims=("time", "units"))
    sliced_data = data[0:5]
    assert sliced_data.dims == data.dims


def test_array_finalize_reduce_dims():
    """
    Test the `__array_finalize__` method for the resetting of the `dims` attributes in
    operations which *reduce* the number of dimensions.

    Operations tested:
    - Slicing to extract a single dimension
    - Reducing method: Summation
    - Reshaping to flatten the data
    - Broadcasting to add a new dimension

    Expected Output
    ---------------
    Default dimensions should be set to the new object and match the new number of dimensions.
    """
    data = CoreData(np.zeros((5, 10)), dims=("time", "units"))
    # Slicing
    sliced_data = data[0]
    assert sliced_data.dims == ("",)
    # Summation
    summed_data = data.sum(axis=0)
    assert summed_data.dims == ("",)  # pylint: disable=no-member
    # Reshaping
    reshaped_data = data.reshape(-1)
    assert reshaped_data.dims == ("",)  # pylint: disable=no-member
    # Broadcasting to add a new dimension
    broadcasted_data = data[:, None]
    assert broadcasted_data.dims == ("", "", "")


def test_use_as_indexer():
    """
    Test whether the core data object can be used as an indexer.

    Types of indexings tested:

    - Integer indexing
    - Condition-based indexing
    """
    # Integer indexing
    values = np.array([1, 2, 3])
    indexer = CoreData(values, dims=("time",))
    array = np.arange(10)
    assert np.array_equal(array[indexer], array[values])
    # Condition-based indexing
    values = np.arange(10)  # same as array
    indexer = CoreData(values, dims=("time",))
    condition_core_data = indexer > 5
    condition_values = values > 5
    assert np.array_equal(array[condition_core_data], array[condition_values])


# TODO: Test `transpose`, `T`, `swapaxes`, `moveaxis`, `rollaxis`
