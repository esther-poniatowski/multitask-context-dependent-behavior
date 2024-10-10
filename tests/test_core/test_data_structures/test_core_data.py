#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`test_core.test_data_structures.core_data` [module]

See Also
--------
`core.data_structures.core_data`: Tested module.
"""

import numpy as np
import pytest

from core.data_structures.core_data import CoreData


@pytest.mark.parametrize("dims", argvalues=[None, ("time", "units")], ids=["default", "with_dims"])
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


def test_delegation():
    """
    Test the `get_axis` and `get_dim` methods delegated to the `Dimensions` class.
    """
    data = CoreData(np.zeros((5, 10)), dims=("time", "units"))
    assert data.get_axis("time") == 0
    assert data.get_dim(0) == "time"


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


def test_as_indexer():
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
