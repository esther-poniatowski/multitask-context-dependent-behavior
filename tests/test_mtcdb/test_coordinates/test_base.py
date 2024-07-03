#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_coordinates.test_base` [module]

Notes
-----
Those tests focus on the functionalities defined in the abstract base class,
and thus common to all coordinate types.
However, a specific subclass is used for actual testing,
because the abstract base class cannot be instantiated.

See Also
--------
:mod:`mtcdb.coordinates.base`: Tested module.
:mod:`mtcdb.coordinates.time`: Tested module.
"""

import numpy as np
import pytest # pylint: disable=unused-import

from mtcdb.coordinates.time import CoordTime


# Dummy coordinate values
N_SMPL = 10
VALUES = np.arange(N_SMPL).astype(np.float64)


def test_len():
    """
    Test :meth:`__len__` which returns the size of the 1D coordinate array.

    Test Inputs
    -----------
    values : np.ndarray
        Coordinate values with shape ``(n_smpl,)``.

    Expected Output
    ---------------
    n_smpl : int
        Number of samples in the coordinate.
    """
    coord = CoordTime(VALUES)
    assert len(coord) == N_SMPL, "Incorrect length."


def test_eq():
    """
    Test :meth:`__eq__` which checks if two coordinates are equal.

    Test Inputs
    -----------
    values : np.ndarray
        Coordinate values with shape ``(n_smpl,)``.

    Expected Output
    ---------------
    True : bool
        The two coordinates are equal.
    False : bool
        The two coordinates are not equal.
    """
    coord1 = CoordTime(VALUES)
    coord2 = CoordTime(VALUES)
    coord3 = CoordTime(VALUES + 1)
    assert coord1 == coord2, "Found Not Equal"
    assert not coord1 == coord3, "Found Equal" # pylint: disable=C0117


def test_copy():
    """
    Test :meth:`copy` which returns a deep copy of the coordinate.

    Test Inputs
    -----------
    values : np.ndarray
        Coordinate values with shape ``(n_smpl,)``.

    Expected Output
    ---------------
    coord : Coordinate
        Deep copy of the input coordinate.
    """
    coord = CoordTime(VALUES)
    coord_copy = coord.copy()
    assert np.array_equal(coord.values, coord_copy.values), "Incorrect copy."


@pytest.mark.parametrize("idx",
                         argvalues=[
                                3,
                                slice(3, 6),
                                (VALUES > 5)
                         ],
                         ids=["int", "slice", "bool"])
def test_getitem(idx):
    """
    Test :meth:`__getitem__` which returns the value of the coordinate at the specified index.

    Test Inputs
    -----------
    values: np.ndarray
        Coordinate values with shape ``(n_smpl,)``.
    idx [int]: int
        Integer index.
    idx [slice]: 
        Slice index.
    idx [bool]: np.ndarray[np.bool_]
        Boolean mask.

    Expected Output
    ---------------
    expected [int]
        Coordinate object with the values at the specified index.
    expected [slice]
        Coordinate object with the values in the specified slice.
    expected [bool]
        Coordinate object with the values filtered by the mask.
    """
    coord = CoordTime(VALUES)
    if isinstance(idx, int):
        expected = VALUES[idx]
    else:
        expected = CoordTime(np.array(VALUES[idx]))
    assert coord[idx] == expected, "Incorrect value at index."
