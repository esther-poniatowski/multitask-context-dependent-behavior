#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_datasets.test_utils` [module]

Tests for the module :mod:`core.data_structures.utils`.

Check that the delegation to the `DataArray` methods and properties
works correctly, and that any custom methods behave as expected.
attribute access, item access, custom methods, and file operations.
"""

import numpy as np
import pytest
import xarray as xr

# from core.data_structures.activity import FiringRates

pytestmark = pytest.mark.skip(reason="Not implemented yet")


def test_attribute_access():
    """Check that accessing attributes like coordinates and dimensions
    proxies correctly to the `DataArray`."""
    # Create a DataArray and a MyCustomData instance
    data = xr.DataArray(np.arange(12).reshape(3, 4), dims=["x", "y"])
    custom_data = MyCustomData(data)
    # Test coordinate access
    np.testing.assert_array_equal(custom_data.coords["x"], np.array([0, 1, 2]))
    np.testing.assert_array_equal(custom_data.coords["y"], np.array([0, 1, 2, 3]))
    # Test dimension access
    assert custom_data.dims == ("x", "y")


def test_item_access():
    """Verifies that slicing and indexing operations are properly delegated to the `DataArray`."""
    data = xr.DataArray(np.arange(12).reshape(3, 4), dims=["x", "y"])
    custom_data = MyCustomData(data)
    # Test slicing
    np.testing.assert_array_equal(custom_data[1, :], np.array([4, 5, 6, 7]))
    np.testing.assert_array_equal(custom_data[:, 1], np.array([1, 5, 9]))


def test_custom_method():
    """Assumes a hypothetical method `custom_mean()` that calculates the mean of the data.
    This test checks if the custom method returns the expected result."""
    data = xr.DataArray(np.random.rand(10), dims=["x"])
    custom_data = MyCustomData(data)
    # Assuming there is a custom method that calculates the mean
    # For illustration, let's add this method to the MyCustomData class:
    # def custom_mean(self):
    #     return self.data_array.mean().item()
    expected_mean = data.mean().item()
    assert custom_data.custom_mean() == expected_mean


def test_save_load(tmp_path):
    """Tests the save and load functionality of the `MyCustomData` class using `pytest`'s `tmp_path` fixture,
    which provides a temporary directory specific to the test session."""
    data = xr.DataArray(np.random.rand(3, 3), dims=["x", "y"])
    custom_data = MyCustomData(data)
    file_path = tmp_path / "test_data.nc"
    # Test saving functionality
    custom_data.save(file_path)
    # Test loading functionality
    loaded_data = MyCustomData.load(file_path)
    xr.testing.assert_identical(loaded_data.data_array, data)
