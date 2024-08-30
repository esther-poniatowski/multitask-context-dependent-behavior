#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_data_structures.test_base` [module]

See Also
--------
:mod:`core.data_structures.base`: Tested module.
:class:`unittest.mock.MagicMock`: Mocking class.
    Principle: Create a mock object to replace the loader method.

"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportRedeclaration=false
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=redefined-outer-name


from types import MappingProxyType
from typing import Dict

import numpy as np
import pytest

from core.data_structures.base import Data


@pytest.fixture
def dim2coord():
    """
    Fixture - Generate the class attribute :attr:`dim2coord`.

    Returns
    -------
    MappingProxyType:
        Two dimensions: 'dim1', 'dim2'.
        One coordinate for 'dim1' and two for 'dim2'.
    """
    return MappingProxyType(
        {"dim1": frozenset(["coord1"]), "dim2": frozenset(["coord2", "coord3"])}
    )


@pytest.fixture
def data(dim2coord) -> np.ndarray:
    """
    Fixture - Generate data consistent with :func:`dim2coord`.

    Returns
    -------
    data: np.ndarray
        Array full of O with length 3 along each dimensions specified in :attr:`dim2coord`.
    """
    n_dims = len(dim2coord)
    shape = tuple(3 for _ in range(n_dims))
    data = np.zeros(shape)
    return data


@pytest.fixture
def coord2type(dim2coord) -> Dict[str, type]:
    """
    Fixture - Generate the class attribute :attr:`coord2type`.

    Returns
    -------
    MappingProxyType:
        Type of each coordinate, here only numpy arrays.

    Notes
    -----
    This is consistent with the use of the method :meth:`empty`, which is available in :mod:`numpy`.
    """
    c2t: Dict[str, type] = {}
    for coord_set in dim2coord.values():
        for coord_name in coord_set:
            c2t[coord_name] = np.ndarray
    return c2t


@pytest.fixture
def coords(dim2coord, data) -> Dict[str, np.ndarray]:
    """
    Fixture - Generate coordinates consistent with :func:`dim2coord` and :func:`data`.

    Returns
    -------
    coords: Dict[str, np.ndarray]
        Coordinates associated to each dimension with appropriate lengths.
        Coordinates are stored in a dictionary with the name of the coordinate as keys, so that they
        can be passed directly as kwargs to the :class:`Data` constructor.
    """
    coords = {}
    for i, (dim, coord_set) in enumerate(dim2coord.items()):
        for coord_name in coord_set:  # all coordinates associated to the dimension
            coords[coord_name] = np.arange(data.shape[i])
    return coords


@pytest.fixture
def subclass(request, tmp_path):
    """
    Fixture - Define a test class inheriting from :class:`Data`.

    Returns
    -------
    TestClass:
        Class inheriting from :class:`Data`. It defines the class attributes :attr:`dim2coord`,
        :attr:`coord2type` and :attr:`path_ruler` required by the metaclass :class:`DataStructMeta`. It
        implements the property :attr:`path` required by the abstract base class :class:`Data`.

    See Also
    --------
    :func:`dim2coord`: Requested fixture for :attr:`dim2coord`.
    :func:`request.getfixturevalue`: Method of the `request` fixture used to retrieve the *value*
        returned by the :func:`dim2coord` fixture. Otherwise, the fixture would be passed as a
        function during attribute assignment, leading to an AttributeError.
    """

    class TestClass(Data):

        dim2coord = request.getfixturevalue("dim2coord")
        coord2type = request.getfixturevalue("coord2type")
        path_ruler = None

        @property
        def path(self):
            return tmp_path

    return TestClass


def test_class_creation(dim2coord, subclass):
    """
    Test subclass creation involving :class:`DataStructMeta` and :class:`Data`.

    Test Inputs
    -----------
    Test class defined with the class attribute :attr:`dim2coord`.

    Expected Output
    ---------------
    Class attributes :attr:`dims`, :attr:`coords` and :attr:`coord2dim` should be set automatically
    and be consistent with the content of :attr:`dim2coord`.

    See Also
    --------
    :func:`dim2coord`: Requested fixture for :attr:`dim2coord`.
    :func:`test_class`: Requested fixture creating a sub-class created via inheritance.
    """
    # Check the presence of the class attributes
    assert hasattr(subclass, "dims")
    assert hasattr(subclass, "coords")
    assert hasattr(subclass, "coord2dim")
    # Check that the dimensions are set correctly
    expected_dims = tuple(dim2coord.keys())
    assert subclass.dims == expected_dims
    # Check that all the coordinates are present in the class attribute
    expected_coords = set([c for coord_set in dim2coord.values() for c in coord_set])
    for c in expected_coords:
        assert c in expected_coords
    # Check that the mapping is correct
    for dim, coord_set in dim2coord.items():
        for c in coord_set:
            assert subclass.coord2dim[c] == dim


@pytest.mark.parametrize("valid", argvalues=[True, False], ids=["Valid", "Invalid"])
def test_init_from_scratch(subclass, data, coords, valid):
    """
    Test :meth:`Data.__init__` to create a data instance from scratch.

    Two possibilities: either with all required coordinates, or with missing coordinates.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Data`.
    data:
        Numpy array which should be passed as the data attribute.
    coords:
        Coordinate arrays which should be passed with the names of the coordinates in the class
        attribute :attr:`coords`.
    valid: bool
        Whether to test with all coordinates or with missing coordinates.

    Expected Output
    ---------------
    [Valid]: All attributes are set correctly.
    [Invalid]: ValueError due to missing coordinates.
    """
    expected_n = MappingProxyType({"dim1": data.shape[0], "dim2": data.shape[1]})
    if valid:
        obj = subclass(data=data, **coords)
        np.testing.assert_array_equal(obj.data, data)
        assert obj.n == expected_n
    else:
        with pytest.raises(ValueError):
            coord_names = list(coords.keys())
            invalid_coords = {k: v for k, v in coords.items() if k != coord_names[0]}  # remove one
            obj = subclass(data=data, **invalid_coords)


def test_getitem(subclass, data, coords):
    """
    Test the :meth:`__getitem__` method to retrieve coordinates by name.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Data`.
    data:
        Numpy array that represents the data to be loaded.
    coords:
        Coordinate arrays that should be part of the loaded data.

    Expected Output
    ---------------
    Coordinates should be correctly retrieved using the bracket notation.
    """
    obj = subclass(data=data, **coords)
    # Test retrieving each coordinate
    for coord_name, coord_value in coords.items():
        assert np.array_equal(
            obj[coord_name], coord_value
        ), f"Coordinate '{coord_name}' not correctly retrieved."
    # Test retrieving a non-existent coordinate
    with pytest.raises(KeyError):
        obj["non_existent_coord"]


def test_save_load(subclass, data, coords):
    """
    Test the :meth:`save` and :meth:`load` methods.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Data`.
    data:
        Numpy array that represents the data to be loaded.
    coords:
        Coordinate arrays that should be part of the loaded data.

    Expected Output
    ---------------
    The object is updated with the loaded data and coordinates.
    """
    # Save an object of the subclass with data and coordinates
    obj = subclass(data=data, **coords)
    obj.save()
    # Load the object in a different instance (same path)
    new_data = np.ones_like(data)
    new_coords = {k: v + 1 for k, v in coords.items()}
    new_obj = subclass(data=new_data, **new_coords)
    new_obj.load()
    # Check that the data and coordinates are the same
    np.testing.assert_array_equal(new_obj.data, data)
    for coord_name in obj.coords:
        np.testing.assert_array_equal(new_obj[coord_name], obj[coord_name])
