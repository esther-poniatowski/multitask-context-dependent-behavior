#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_coordinates.test_exp_structure` [module]

See Also
--------
:mod:`core.coordinates.exp_structure`: Tested module.

Notes
-----
Those tests focus on the functionalities which are *specific* to
one concrete subclass of coordinates.

The tests are performed on the :class:`CoordRecording`subclass,
but apply for all subclasses of :class:`CoordExpStructure`.
"""

import numpy as np
import pytest

from core.coordinates.exp_structure_coord import CoordRecording
from core.attributes.exp_structure import Recording


N_SMPL = 10


def test_coord_recording_build_labels():
    """
    Test :meth:`CoordRecording.build_labels`.

    Test Inputs
    -----------
    N_SMPL: 10
    rec: Recording(1)

    Expected Output
    ---------------
    values: 10 samples of 1.
    """
    values = CoordRecording.build_labels(n_smpl=N_SMPL, pos=Recording(1))
    expected_values = np.full(N_SMPL, 1, dtype=np.int64)
    assert np.array_equal(values, expected_values)


def test_coord_recording_count_by_lab():
    """
    Test :meth:`CoordRecording.count_by_lab`.

    Test Inputs
    -----------
    initial_values : 5 samples of 1, 5 samples of 2.

    Expected Output
    ---------------
    count : Two Recording objects as keys, with count 5 for each.
    """
    values = np.array(5 * [1] + 5 * [2])
    coord = CoordRecording(values=values)
    count = coord.count_by_lab()
    expected_count = {1: 5, 2: 5}
    assert count == expected_count
