#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`test_core.test_coordinates.test_time` [module]

See Also
--------
`core.coordinates.time`: Tested module.

Notes
-----
Those tests focus on the functionalities which are *specific* to one concrete subclass of coordinates.
"""

import numpy as np
import pytest

from core.coordinates.time_coord import CoordTime


N_SMPL = 10
T_BIN = 0.1
VALUES = np.arange(N_SMPL).astype(np.float64)


def test_coord_time_set_t_bin():
    """
    Test :meth:`CoordTime.set_t_bin` which sets the time bin of the coordinate.

    Test Inputs
    -----------
    values : np.ndarray
        Coordinate values with bin size of 0.1.

    Expected Output
    ---------------
    t_bin : float
        Time bin of the coordinate.
    """
    values = np.arange(N_SMPL) * T_BIN
    coord_time = CoordTime(values=values)
    coord_time.set_t_bin()
    assert coord_time.t_bin == 0.1


@pytest.mark.parametrize(
    "n_smpl, t_bin, t_max",
    argvalues=[
        (N_SMPL, T_BIN, None),
        (N_SMPL, None, 1.0),
        (None, T_BIN, 1.0),
    ],
    ids=[
        "n_smpl, t_bin",
        "n_smpl, t_max",
        "t_bin, t_max",
    ],
)
def test_coord_time_build_labels_valid(n_smpl, t_bin, t_max):
    """
    Test :meth:`CoordTime.build_labels` with a valid combination of parameters.

    Test Inputs
    -----------
    n_smpl : 10 (when not None)
    t_bin : 0.1 (when not None)
    t_max : 1.0 (when not None)

    Expected Output
    ---------------
    expected_values : np.ndarray
        In each case, it should contain
        10 time points at 0.1 s bin between 0 and 1.
    """
    expected_values = np.arange(N_SMPL) * T_BIN
    values = CoordTime.build_labels(n_smpl=n_smpl, t_bin=t_bin, t_max=t_max)
    assert np.array_equal(values, expected_values)


def test_coord_time_build_labels_invalid():
    """
    Test :meth:`CoordTime.build_labels` with invalid combination of parameters.

    Test Inputs
    -----------
    n_smpl, t_bin, t_max provided together.

    Expected Output
    ---------------
    ValueError
    """
    with pytest.raises(ValueError):
        CoordTime.build_labels(n_smpl=10, t_bin=0.1, t_max=1.0)
