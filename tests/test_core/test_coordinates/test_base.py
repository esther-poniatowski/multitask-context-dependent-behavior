#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`test_core.test_coordinates.test_base` [module]

Notes
-----
Those tests focus on the functionalities defined in the abstract base class, which are common to all
coordinate types. However, a *specific* subclass is used for actual testing (:class:`CoordTime`),
because the abstract base class cannot be instantiated.

See Also
--------
`core.coordinates.base_coordinate`: Tested module.
`core.coordinates.time`: Tested module.
"""

import numpy as np
import pytest

from core.coordinates.time_coord import CoordTime


N_SMPL = 10
"""Number of samples in the test values. Defined globally for consistency in parametrized tests."""


@pytest.fixture
def values():
    """
    Generate dummy coordinate values used for testing.

    Returns
    -------
    np.ndarray
        Coordinate values with shape ``(n_smpl,)``.
    """

    return np.arange(N_SMPL).astype(np.float64)


def test_len(values):
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
    n_smpl = len(values)
    coord = CoordTime(values)
    assert len(coord) == n_smpl, f"Incorrect length: expected {n_smpl}, got {len(coord)}."


def test_eq(values):
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
    coord1 = CoordTime(values)
    coord2 = CoordTime(values)
    coord3 = CoordTime(values + 1)
    assert coord1 == coord2, "Found Not Equal"
    assert not coord1 == coord3, "Found Equal"


def test_copy(values):
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
    coord = CoordTime(values)
    coord_copy = coord.copy()
    assert np.array_equal(coord.values, coord_copy.values), "Incorrect copy."


@pytest.mark.parametrize(
    "idx", argvalues=[3, slice(3, 6), (np.arange(N_SMPL) > 5)], ids=["int", "slice", "bool"]
)
def test_getitem(values, idx):
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
    coord = CoordTime(values)
    if isinstance(idx, int):
        expected = values[idx]
    else:
        expected = CoordTime(np.array(values[idx]))
    assert coord[idx] == expected, "Incorrect value at index."


def test_empty():
    """
    Test :meth:`empty` which returns an empty coordinate.

    Expected Output
    ---------------
    coord : Coordinate
        Empty coordinate of the subclass type.
    """
    coord = CoordTime.empty()
    assert isinstance(coord, CoordTime), "Incorrect type."
    assert len(coord) == 0, "Non-empty coordinate."
