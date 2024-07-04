#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_data_structures.test_base` [module]

See Also
--------
:mod:`mtcdb.data_structures.base`: Tested module.
"""
# pylint: disable=missing-class-docstring
# pylint: disable=no-member

from types import MappingProxyType

import numpy as np
import pytest # pylint: disable=unused-import

from mtcdb.data_structures.base import MetaData, Data


# Dimensions and Coordinates for Test Subclass
DIM2COORD = MappingProxyType({
    'dim1': frozenset(['coord1']),
    'dim2': frozenset(['coord1', 'coord2'])
})


@pytest.mark.parametrize("cls, cls_type",
                         argvalues=[
                             ("TestData1", "MetaData"),
                             ("TestData2", "Data")
                        ],
                        ids=["MetaData", "Data"])
def test_meta_class(cls, cls_type):
    """
    Test :class:`MetaData` and inheritance from :class:`Data`.

    Test Inputs
    -----------
    Test classes with the class attribute :attr:`dim2coord`.
    Two dimensions, 'dim1' and 'dim2'.
    One coordinate for 'dim1' and two for 'dim2'.
    TestData [MetaData]: 
        Using the metaclass :class:`MetaData`.
    TestData2 [Data]:
        Using inheritance from :class:`Data`.

    Expected Output
    ---------------
    Creation of the attributes :attr:`dims`, :attr:`coords` and :attr:`coord2dim`.
    """
    # Dynamically create the classes for testing
    if cls_type == "MetaData":
        class TestClass(metaclass=MetaData): # type: ignore
            dim2coord = DIM2COORD
    elif cls_type == "Data":
        class TestClass(Data):
            dim2coord = DIM2COORD
            @property
            def path(self):
                return 'path/to/data'
    expected_dims = ('dim1', 'dim2')
    expected_coords = ('coord1', 'coord2')
    expected_coord2dim = MappingProxyType({
        'coord1': 'dim1',
        'coord2': 'dim2'
    })
    assert TestClass.dims == expected_dims  # type: ignore
    assert TestClass.coords == expected_coords  # type: ignore
    assert TestClass.coord2dim == expected_coord2dim  # type: ignore


@pytest.mark.parametrize("valid", [True, False], ids=["Valid", "Invalid"])
def test_base_init(valid):
    """
    Test :meth:`Data.__init__` with all required coordinates.

    Test Inputs
    -----------
    TestData:
        Class inheriting from :class:`Data`.
    data: 
        Numpy array.
    **kwargs: 
        Coordinate arrays passed with the names
        of the coordinates in the class attribute :attr:`coords`.
    """
    class TestData(Data):
        """Test class"""
        dim2coord = DIM2COORD
        @property
        def path(self):
            return 'path/to/data'
    data = np.zeros((3, 3, 3))
    coord1 = np.ones(data.shape[0])
    coord2 = np.ones(data.shape[1])
    if valid:
        obj = TestData(data=data, coord1=coord1, coord2=coord2)
        np.testing.assert_array_equal(obj.data, data)
        np.testing.assert_array_equal(obj.coord1, coord1)  # type: ignore
        np.testing.assert_array_equal(obj.coord2, coord2)  # type: ignore
    else:
        with pytest.raises(ValueError):
            TestData(data=data, coord2=coord2)
