#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_pipelines.test_preprocess.test_firing_rates` [module]

Tests for the module :mod:`core.processors.preprocess.firing_rates`.

See Also
--------
np.testing.assert_array_almost_equal:
    Assess element-wise equality between 2 arrays within a tolerance. Used to ignore numerical
    errors due to floating point arithmetic. Specifically designed for testing purposes (detailed
    error messages). Default tolerance: 1e-5.
"""
# pylint: disable=expression-not-assigned

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_array_eq
import pytest

from core.processors.preprocess.align_spikes import SpikesAligner


def test_slice_epoch():
    """
    Test for :meth:`slice_epoch`.

    Test Inputs
    -----------
    spk  : :obj:`core.types.NumpyArray`
        Spikes evenly distributed in ``[0, 1]`` every 0.1 s (10 spikes), i.e.:
        ``[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``
    t_start = 0.2
    t_end = 0.5

    Expected Outputs
    ----------------
    expected [with_content]: ``[0.0, 0.1, 0.2]``
        First, 3 spikes are retained in the epoch, originally at ``[0.2, 0.3, 0.4]`` sec (since the
        starting time is included but not the ending time). Then, spiking times are shifted by the
        start time of the epoch, which requires to subtract 0.2 to each value.
    """
    spk = np.arange(0, 1, 0.1)
    expected = np.array([0.0, 0.1, 0.2])
    t_start, t_end = 0.2, 0.5
    aligner = SpikesAligner()
    aligner.spikes = spk  # set manually
    sliced = aligner.slice_epoch(t_start, t_end)
    assert sliced.shape == expected.shape, f"Output shape: {sliced.shape} != {expected.shape}"
    assert_array_eq(sliced, expected), f"Output values: {sliced} != {expected}"


spk_0to2 = np.arange(0, 2, 0.1)
expected_0to2 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
spk_1to2 = np.arange(1, 2, 0.1)
expected_1to2 = np.array([0.4, 0.5, 0.6, 0.7])
t_start1, t_end1 = 0.3, 0.7
t_start2, t_end2 = 1.3, 1.7


@pytest.mark.parametrize(
    "spk, expected",
    argvalues=[(spk_0to2, expected_0to2), (spk_1to2, expected_1to2), (np.array([]), np.array([]))],
    ids=["0to2", "1to2", "empty"],
)
def test_join_epochs(spk, expected):
    """
    Test for :meth:`join_epochs`.

    Test Inputs
    -----------
    spk [0to2]: :obj:`core.types.NumpyArray`
        Spikes evenly distributed in ``[0, 2]`` every 0.1 s (20 spikes).
    spk [1to2]: :obj:`core.types.NumpyArray`
        Spikes only after 1 s, so that there is no spike in the first epoch.
    spk [empty]: :obj:`core.types.NumpyArray`
        No spikes at all (empty array).
    t_start1, t_end1, t_start2, t_end2: float
        Two epochs of 0.4 s duration : [0.3, 0.7] and [1.3, 1.7].

    Expected Outputs
    ----------------
    expected [0to2]: :obj:`core.types.NumpyArray`
        ``[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]``
        3 spikes are retained in each epoch, originally at
        ``[0.3, 0.4, 0.5, 0.6]`` and ``[1.3, 1.4, 1.5, 1.6]``
        Then, the spiking times of both epoch are shifted by 0.3 and 1.3 respectively:
        ``[0.0, 0.1, 0.2, 0.3]`` and ``[0.0, 0.1, 0.2, 0.3]``
        Finally, the second epoch is shifted  the duration of the first epoch
        (0.4 s) to be contiguous with the first epoch, hence the result.
    expected [1to2]: :obj:`core.types.NumpyArray`
        Spikes in the second epoch should still be retained
        and shifted at ``[0.4, 0.5, 0.6, 0.7]``.
    expected [empty]: :obj:`core.types.NumpyArray`
        No spike should be retained.
    """
    aligner = SpikesAligner()
    aligner.spikes = spk  # set manually
    spk_joined = aligner.join_epochs(t_start1, t_end1, t_start2, t_end2)
    shape = spk_joined.shape
    assert shape == expected.shape, f"Output shape: {shape} != {expected.shape}"
    assert_array_eq(spk_joined, expected), "Wrong values"
