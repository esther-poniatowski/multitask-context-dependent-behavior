#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`test_core.test_data_structures.test_base` [module]

See Also
--------
`core.data_structures.base_data_structure`: Tested module.
:class:`unittest.mock.MagicMock`: Mocking class.
    Principle: Create a mock object to replace the loader method.

"""
# pylint: disable=protected-access


# pyright: reportAttributeAccessIssue=false
# pyright: reportRedeclaration=false
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=redefined-outer-name


from types import MappingProxyType
from typing import Dict, Tuple, Mapping

import numpy as np
import pytest

from core.data_structures.base_data_structure import DataStructure, LazyAttribute
from core.data_components.core_data import Dimensions, CoreData

# --- Fixture to Define a Subclass -----------------------------------------------------------------


@pytest.fixture
def dims_attr() -> Dimensions:
    """
    Fixture - Generate the class attribute `dims` for the test subclass.

    Returns
    -------
    dims : Tuple[DimName, ...]
    """
    return Dimensions("time", "trials")


@pytest.fixture
def coords_attr() -> Mapping[str, type]:
    """
    Fixture - Generate the class attribute `coords` for the test subclass.

    Returns
    -------
    MappingProxyType:
        Type of each coordinate (here only numpy arrays for simplicity).
    """
    return MappingProxyType({"time": np.ndarray, "task": np.ndarray, "categ": np.ndarray})


@pytest.fixture
def coords_to_dims() -> Mapping[str, Dimensions]:
    """
    Fixture - Generate the class attribute `coords_to_dims` for the test subclass.

    Returns
    -------
    coords_to_dims : MappingProxyType
        Mapping between coordinates and their associated dimensions.
    """
    return MappingProxyType(
        {"time": Dimensions("time"), "task": Dimensions("trials"), "categ": Dimensions("trials")}
    )


@pytest.fixture
def subclass(request, dims_attr, coords_attr, coords_to_dims, tmp_path):
    """
    Fixture - Define a test class inheriting from the base `DataStructure` class.

    - Define the required class attributes: `dims`, `coords`, `coords_to_dims`, `identifiers`.
    - Implement the `path` property to return a temporary directory path for testing file I/O
      operations.

    Returns
    -------
    TestClass:
        Class inheriting from `DataStructure`.

    See Also
    --------
    :func:`request.getfixturevalue`: Method of the `request` fixture used to retrieve the *value*
        returned by fixtures which define class attributes. Otherwise, the fixture would be passed
        as a function during attribute assignment, leading to an AttributeError.
    :func:`tmp_path`: Fixture to create a temporary directory for testing file I/O operations.
    """

    class TestClass(DataStructure):
        dims = Dimensions(request.getfixturevalue("dims_attr"))
        coords = MappingProxyType(request.getfixturevalue("coords_attr"))
        coords_to_dims = MappingProxyType(request.getfixturevalue("coords_to_dims"))
        identifiers = ("id",)

        @property
        def path(self):
            return tmp_path

    return TestClass


@pytest.fixture
def subclass_missing_attr(request, dims_attr, coords_attr, coords_to_dims):
    """
    Fixture - Define an invalid test class inheriting from the base `DataStructure` class.

    Missing required class attribute: `identifiers`.

    Returns
    -------
    create_invalid_subclass : function
        Function which creates the invalid subclass. This is necessary to avoid raising an error at
        the fixture definition, since the `__init_subclass__` method is called when the subclass is
        *defined* (not when it is *instantiated*).
    """

    def create_invalid_subclass():
        class TestClass(DataStructure):
            dims = Dimensions(request.getfixturevalue("dims_attr"))
            coords = MappingProxyType(request.getfixturevalue("coords_attr"))
            coords_to_dims = MappingProxyType(request.getfixturevalue("coords_to_dims"))

        return TestClass

    return create_invalid_subclass


@pytest.fixture
def valid_data(subclass):
    """
    Fixture - Generate valid data values for the test subclass.

    Returns
    -------
    data_values : np.ndarray
        Array of zeros with the same number of dimensions as the subclass.
    """
    N = 10  # number of elements along each dimension
    return np.zeros(tuple(N for _ in range(len(subclass.dims))))


@pytest.fixture
def valid_coord(subclass, valid_data):
    """
    Fixture - Generate valid coordinate values for the first coordinate of the test subclass.

    Returns
    -------
    coord_name : str
        Name of the first coordinate.
    coord_values : np.ndarray
        Array of zeros with the same number of dimensions as the data in the subclass.
    """
    N = valid_data.shape[0]  # number of elements (identical along all dimensions)
    coord_name = list(subclass.coords.keys())[0]  # first coordinate
    coord_values = np.zeros(tuple(N for _ in subclass.coords_to_dims[coord_name]))
    return coord_name, coord_values


# --- Test Subclass Creation -----------------------------------------------------------------------


def test_init_subclass_invalid(subclass_missing_attr):
    """
    Test for the `__init_subclass__` method: Detection of missing class attributes whe an invalid
    subclass is defined.

    Expected Output
    ---------------
    TypeError, since the created subclass is missing the required class attribute `identifiers`.
    """
    with pytest.raises(TypeError):
        subclass_missing_attr()  # call `create_invalid_subclass` to create the invalid subclass


def test_init_subclass_lazy_attributes(subclass, dims_attr, coords_attr):
    """
    Test for the `__init_subclass__` method: Setting up of lazy attributes.

    Expected Output
    ---------------
    Content-related attributes should be instances of the `LazyAttribute` descriptor: `data` and
    each coordinate name in `coords`.
    """
    # Check if 'data' is a LazyAttribute
    assert isinstance(subclass.__dict__["data"], LazyAttribute)
    # Check if all coordinates are LazyAttributes
    for coord_name in subclass.coords:
        assert isinstance(subclass.__dict__[coord_name], LazyAttribute)


def test_init_empty_attributes(subclass):
    """
    Test the constructor: Default initialization of the hidden representation of the content-related
    attributes to `None`.

    Expected Outputs
    ----------------
    The hidden attributes `_data` and each coordinate name prefixed with an underscore should be set
    to `None`.
    Accessing one of these attributes with a public name should raise an AttributeError.
    """
    instance = subclass()
    assert instance._data is None
    for coord_name in instance.coords:
        assert getattr(instance, f"_{coord_name}") is None
    with pytest.raises(AttributeError):
        _ = instance.data


def test_set_data(subclass, valid_data):
    """
    Test the setter method `set_data` for the data attribute.

    Test Inputs
    -----------
    valid_data :
        Numpy array with the same number of dimensions as the subclass.

    Expected Output
    ---------------
    - Setting the `data` attribute via the dedicated setter method should update the hidden
      attribute `_data`.
    - The content should be automatically converted to a CoreData object with the dimensions
      specified in the subclass class-attribute `dims`.
    - Setting data values with an invalid shape should raise a ValueError.
    - Setting the `data` attribute via direct assignment should raise an AttributeError,
      since no setter method is provided by the `LazyAttribute` descriptor. TODO: Not the case.
    """
    instance = subclass()
    # Set valid data values
    instance.set_data(valid_data)
    assert isinstance(instance._data, CoreData)
    # Stack the valid_data array to get an invalid shape
    invalid_data = np.stack([valid_data, valid_data], axis=0)
    with pytest.raises(ValueError):
        instance.set_data(invalid_data)


def test_set_coords(subclass, valid_data, valid_coord):
    """
    Test the setter method `set_coords` for coordinate attributes.

    Expected Output
    ---------------
    - Setting coordinate attributes via the dedicated setter method should update the hidden
      attributes `_coord_name`.
    - The content should be automatically converted to the expected Coordinate type.
    - Invalid coordinate names or shapes should raise ValueError.
    """
    # Set valid data values
    # Use the setter to convert it to CoreData (required to check for valid coord shape)
    instance = subclass()
    instance.set_data(valid_data)
    # Set one valid coordinate
    coord_name, coord_values = valid_coord
    instance.set_coords(**{coord_name: coord_values})
    expected_coord_type = subclass.coords[coord_name]
    assert isinstance(getattr(instance, f"_{coord_name}"), expected_coord_type)
    # Test invalid coordinate name
    with pytest.raises(ValueError):
        instance.set_coords(invalid_name=np.array([1, 2, 3]))
    # Test invalid coordinate shape
    invalid_coord_values = np.stack([coord_values, coord_values], axis=0)
    with pytest.raises(ValueError):
        instance.set_coords(**{coord_name: invalid_coord_values})


def test_get_coord(subclass, valid_coord):
    """
    Test the getter method `get_coords` for coordinate attributes.

    Expected Output
    ---------------
    - Getting coordinate attributes should return the content of the hidden attributes `_coord_name`.
    - Accessing a coordinate that has not been set should raise an AttributeError.
    """
    instance = subclass()
    # Set one coordinate manually
    coord_name, coord_values = valid_coord
    setattr(instance, f"_{coord_name}", coord_values)
    # Test getting the coordinate
    assert np.array_equal(instance.get_coord(coord_name), coord_values)
    # Test getting a non-existent coordinate
    with pytest.raises(AttributeError):
        _ = instance.get_coord("non_existent_coord")


def test_delegation(subclass, valid_data):
    """
    Test the delegation of several methods to nested attributes via `__getattr__`.
    """
    # Set valid data values (required for delegation to the data attribute)
    instance = subclass()
    instance.set_data(valid_data)
    assert instance.get_axis("time") == 0
    assert instance.get_dim(0) == "time"
    assert instance.get_size("time") == valid_data.shape[0]


def test_save_load(subclass, valid_data, valid_coord):
    """
    Test the `save` and `load` methods.

    Expected Output
    ---------------
    The object is updated with the loaded data and coordinates.
    """
    # Save an object of the subclass with data and coordinates
    coords = {valid_coord[0]: valid_coord[1]}
    obj = subclass(data=valid_data, **coords)
    obj.save()
    # Load the object in a different instance (same path)
    new_data = np.ones_like(valid_data)
    new_coords = {k: v + 1 for k, v in coords.items()}  # add 1 to the coordinate values
    new_obj = subclass(data=new_data, **new_coords)
    new_obj.load()
    # Check that the data and coordinates are the same
    np.testing.assert_array_equal(new_obj.data, obj.data)
    for coord_name in obj.coords:
        np.testing.assert_array_equal(new_obj.get_coord(coord_name), obj.get_coord(coord_name))
