#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_preprocess.test_firing_rates` [module]
============================================================

Tests for the module :mod:`mtcdb.preprocess.firing_rates`.
"""

import numpy as np
import pytest

from mtcdb.preprocess.firing_rates import spikes_to_rates, smooth


def test_bin_spikes():
    """
    Test for :func:`mtcdb.preprocess.firing_rates.spikes_to_rates`.
    
    Test Inputs
    -----------
    tmax: float
        Set to 1.0, for a recording period spanning ``[0, 1]``.
    tbin: float
        Set to 0.1, to divide the recording period into 10 bins.
    spk: numpy.ndarray
        20 spikes evenly distributed in the recoding period.
    
    Expected Outputs
    ----------------
    expected: NumpyArray
        10 values (10 bins) of 20 spikes/s.
        Shape: ``(1, 10)`` (1 trial, 10 bins).
	
    See Also
    --------
    numpy.allclose: 
        Assess element-wise equality between 2 arrays within a tolerance.
        Used to ignore numerical errors due to floating point arithmetic.
	"""
    tmax = 1.0
    tbin = 0.1
    spk = np.linspace(0, tmax, num=20)
    expected = np.full(shape=(10, 1), fill_value=20.0)
    frates = spikes_to_rates(spk, tbin, tmax)
    assert frates.shape == expected.shape, "Wrong shape"
    assert np.allclose(frates, expected), "Wrong values"



def test_smooth():
    """
    Test for :func:`mtcdb.preprocess.firing_rates.smooth`.

    Test Inputs
    -----------
    frates: NumpyArray
        Shape: ``(10, 2)`` (10 bins, 2 trials).
        Values: Two alternating patterns of 0 and 1, repeated 5 times (10 values).
        Both patterns are similar but inverted (0, 1 vs. 1, 0).
    tbin: float
        Set to 0.1, for a recording period of ``10 * 0.1 = 1`` s.
    window: int
        Set to 0.2, for a boxcar kernel of 2 bins width.
    mode: str
        Set to ``'valid'``, to keep only the values which are not
        influenced by zero-padding.

    Expected Outputs
    ----------------
    expected: NumpyArray
        Shape: ``(7, 2)`` (7 time bins, 2 trials).
        In the ``'valid'`` mode, the output time dimension is given by:
        ``ntpts_out = ntpts - window/tbin + 1``.
        Here: ``window/tbin = 2`` bins, so ``ntpts_out = 10 - 2 + 1 = 9``.
        Output values: All equal to 0.5, 
        which is the average of two consecutive values in the input (0 and 1),
        whatever their order in the patterns.
    
    See Also
    --------
    numpy.allclose
    numpy.tile
    """
    frates = np.tile(np.array([[0, 1], [1, 0]]), (5, 1)) # shape: (10, 2)
    tbin = 0.1
    window = 0.2
    mode = 'valid'
    expected = np.full(shape=(9, 2), fill_value=0.5)
    smoothed = smooth(frates, window, tbin, mode)
    assert smoothed.shape == expected.shape, "Wrong shape"
    assert np.allclose(smoothed, expected), "Wrong values"
