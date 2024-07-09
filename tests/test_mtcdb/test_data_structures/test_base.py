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

from mtcdb.data_structures.base import MetaData, Data

@pytest.fixture
def test_dim2coord():
    """
    Fixture for the class attribute :attr:`dim2coord`.
    
    Returns
    -------
    MappingProxyType:
        Two dimensions: 'dim1', 'dim2'.
        One coordinate for 'dim1' and two for 'dim2'.
    """
    return MappingProxyType({
        'dim1': frozenset(['coord1']),
        'dim2': frozenset(['coord2', 'coord3'])
    })

@pytest.fixture
def test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixture for data and coordinates, consistent with :func:`dim2coord`.

    Returns
    -------
    data: np.ndarray
        Data array with two dimensions.
    coord1, coord2, coord3: np.ndarray
        Coordinate arrays for 'dim1' and 'dim2', with appropriate lengths.
    """
    shape = (3, 3)
    data = np.zeros(shape)
    coord1 = np.ones(shape[0])
    coord2 = np.ones(shape[1])
    coord3 = np.ones(shape[1])
    return data, coord1, coord2, coord3

@pytest.fixture
def test_class(test_dim2coord):
    """
    Fixture which creates a test class inheriting from :class:`Data`.

    Returns
    -------
    TestClass:
        Class inheriting from :class:`Data`.
        It has the class attribute :attr:`dim2coord`.
        It implements the property :attr:`path` 
        which is required by the abstract base class.
    
    See Also
    --------
    :func:`dim2coord`: Requested fixture for :attr:`dim2coord`.
    """
    class TestClass(Data):
        dim2coord = test_dim2coord

        @property
        def path(self):
            return 'path/to/data'

    return TestClass


@pytest.mark.parametrize("approach",
                        argvalues=["MetaClass", "BaseClass"],
                        ids=["MetaClass", "BaseClass"])
def test_class_creation(approach, test_dim2coord, test_class):
    """
    Test :class:`MetaData` and inheritance from :class:`Data`.

    Test Inputs
    -----------
    Test classes with the class attribute :attr:`dim2coord`.
    TestClass [MetaData]: 
        Using the metaclass :class:`MetaData`.
    TestClass [Data]:
        Using inheritance from :class:`Data`.
    
    Expected Output
    ---------------
    Creation of the attributes :attr:`dims`, :attr:`coords` and :attr:`coord2dim`.
    Those objects should be consistent with the definition of :attr:`dim2coord`.

    See Also
    --------
    :func:`dim2coord`: Requested fixture for :attr:`dim2coord`.
    :func:`test_class`: Requested fixture a class created via the base class approach.
    """
    # Dynamically create the classes for testing
    if approach == "MetaClass": # override the test class
        class TestClass(metaclass=MetaData):
            dim2coord = test_dim2coord
    elif approach == "BaseClass": # already created by the fixture
        TestClass = test_class
    expected_dims = ('dim1', 'dim2')
    expected_coords = ('coord1', 'coord2', 'coord3')
    expected_coord2dim = MappingProxyType({
        'coord1': 'dim1',
        'coord2': 'dim2',
        'coord3': 'dim2',
    })
    assert TestClass.dims == expected_dims
    assert TestClass.coords == expected_coords
    assert TestClass.coord2dim == expected_coord2dim


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
    expected_n = MappingProxyType({'dim1': data.shape[0], 'dim2': data.shape[1]})
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
    dim2coord = MappingProxyType({
        'dim1': frozenset(['coord_x']),
        'dim2': frozenset(['coord_y'])
    })
    def __init__(self, data=None, **kwargs):
        if data is None:
            data = np.zeros((len(kwargs['coord_x']), len(kwargs['coord_y'])))
        super().__init__(data, **kwargs)
    @property
    def path(self):
        return 'path/to/data'

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
    