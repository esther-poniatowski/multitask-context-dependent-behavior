#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_data_structures.test_base` [module]

See Also
--------
:mod:`mtcdb.data_structures.base`: Tested module.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportRedeclaration=false

from types import MappingProxyType
from typing import Tuple, Mapping, FrozenSet

import numpy as np
import pytest

from mtcdb.data_structures.base import Data


@pytest.fixture
def dim2coord():
    """
    Fixture to generate the class attribute :attr:`dim2coord`.

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
    Fixture to generate data consistent with :func:`dim2coord`.

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
def coords(dim2coord, data) -> Tuple[np.ndarray, ...]:
    """
    Fixture to generate coordinates consistent with :func:`dim2coord` and :func:`data`.

    Returns
    -------
    coord1, coord2...: np.ndarray
        Arrays associated to each dimension, with appropriate lengths.
    """
    coords = []
    for i, (dim, coord_set) in enumerate(dim2coord.items()):
        for c in coord_set:  # all coordinates associated to the dimension
            coords.append(np.arange(data.shape[i]))
    return tuple(coords)


@pytest.fixture
def subclass(request):
    """
    Fixture to create a test class inheriting from :class:`Data`.

    Returns
    -------
    TestClass:
        Class inheriting from :class:`Data`.
        It defines the class attribute :attr:`dim2coord` required by the metaclass :class:`MetaData`.
        It implements the property :attr:`path` required by the abstract base class :class:`Data`.

    See Also
    --------
    :func:`dim2coord`: Requested fixture for :attr:`dim2coord`.
    :func:`request.getfixturevalue`: Method of the `request` fixture used to retrieve the *value*
        returned by the :func:`dim2coord` fixture. Otherwise, the fixture would be passed as a
        function during attribute assignment, leading to an AttributeError.
    """

    class TestClass(Data):
        dim2coord = request.getfixturevalue("dim2coord")

        @property
        def path(self):
            return "path/to/data"

    return TestClass


def test_class_creation(dim2coord, subclass):
    """
    Test subclass creation involving :class:`MetaData` and :class:`Data`.

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
def test_base_init(valid, test_class, test_data):
    """
    Test :meth:`Data.__init__` with all required coordinates or missing coordinates.

    Test Inputs
    -----------
    TestData:
        Class inheriting from :class:`Data`.
    data:
        Numpy array.
    **kwargs:
        Coordinate arrays passed with the names
        of the coordinates in the class attribute :attr:`coords`.

    Expected Output
    ---------------
    [Valid]: All attributes are set correctly.
    [Invalid]: ValueError due to missing coordinates.
    """
    TestClass = test_class
    data, coord1, coord2, coord3 = test_data
    expected_n = MappingProxyType({"dim1": data.shape[0], "dim2": data.shape[1]})
    if valid:
        obj = TestClass(data=data, coord1=coord1, coord2=coord2, coord3=coord3)
        np.testing.assert_array_equal(obj.data, data)
        np.testing.assert_array_equal(obj.coord1, coord1)
        np.testing.assert_array_equal(obj.coord2, coord2)
        assert obj.n == expected_n
    else:
        with pytest.raises(ValueError):
            TestClass(data=data, coord1=coord1)


# ----------------------------------------------------------------------------
class TestData(Data):
    dim2coord = MappingProxyType({"dim1": frozenset(["coord_x"]), "dim2": frozenset(["coord_y"])})

    def __init__(self, data=None, **kwargs):
        if data is None:
            data = np.zeros((len(kwargs["coord_x"]), len(kwargs["coord_y"])))
        super().__init__(data, **kwargs)

    @property
    def path(self):
        return "path/to/data"


def test_init_two_ways():
    """
    Test for the initialization of a :class:`Data` subclass in two ways.

    Test Inputs
    -----------
    TestData:
        Class inheriting from :class:`Data`.
        It implements its own :meth:`__init__` method.
        If data is provided, it calls the base class :meth:`__init__`.
    """
