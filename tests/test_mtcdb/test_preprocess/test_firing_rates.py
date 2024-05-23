#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_preprocess.test_firing_rates` [module]
============================================================

Tests for the module :mod:`mtcdb.preprocess.firing_rates`.

See Also
--------
np.testing.assert_array_almost_equal: 
    Assess element-wise equality between 2 arrays within a tolerance.
    Used to ignore numerical errors due to floating point arithmetic.
    Designed specifically for testing purposes (detailed error messages).
    Default tolerance: 1e-5.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_array_eq
import pytest

from mtcdb.preprocess.firing_rates import extract_trial
from mtcdb.preprocess.firing_rates import slice_epoch
from mtcdb.preprocess.firing_rates import spikes_to_rates
from mtcdb.preprocess.firing_rates import smooth


def test_extract_trial():
    """
    Test for :func:`mtcdb.preprocess.firing_rates.extract_trial`.
    
    Test Inputs
    -----------
    trial: int
        Set to 2, to extract the spiking times of the second trial.
    data: NumpyArray
        3 trials with 2 spikes each.
    """
    data = np.array([
        [1, 1, 2, 2, 3, 3],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ])
    expected = np.array([0.3, 0.4])
    spk = extract_trial(trial=2, data=data)
    assert spk.shape == expected.shape, "Wrong shape"
    assert_array_eq(spk, expected), "Wrong values"


def test_slice_epoch():
    """
    Test for :func:`mtcdb.preprocess.firing_rates.slice_epoch`.
    
    Test Inputs
    -----------
    t0: float
        Set to 0.2, as the start time of the epoch.
    t1: float
        Set to 0.5, as the end time of the epoch.
    spk: numpy.ndarray
        10 spikes evenly distributed in [0, 1], i.e. 1 spike every 0.1 s.
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    Expected Outputs
    ----------------
    expected: NumpyArray
        ``[0.0, 0.1, 0.2]``
        The retained 3 spikes originally occur at ``[0.2, 0.3, 0.4]``, 
        since the starting time is included but not the ending time.
        Then, the times are shifted by the start time of the epoch, i.e. 0.2.
    """
    t0 = 0.2
    t1 = 0.5
    spk = np.arange(0, 1, 0.1)
    expected = np.array([0.0, 0.1, 0.2])
    sliced = slice_epoch(t0, t1, spk)
    assert sliced.shape == expected.shape, "Wrong shape"
    assert_array_eq(sliced, expected), "Wrong values"


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
	"""
    tmax = 1.0
    tbin = 0.1
    spk = np.linspace(0, tmax, num=20)
    expected = np.full(shape=(10, 1), fill_value=20.0)
    frates = spikes_to_rates(spk, tbin, tmax)
    assert frates.shape == expected.shape, "Wrong shape"
    assert_array_eq(frates, expected), "Wrong values"


def test_smooth():
    """
    Test for :func:`mtcdb.preprocess.firing_rates.smooth`.

    Test Inputs
    -----------
    frates: NumpyArray
        Shape: ``(10, 2)`` (10 bins, 2 trials).
        Values: Two alternating patterns of two values, repeated 5 times (10 values).
        Trial 1 : 0, 1. Trial 2 : 0.25, 0.75.
        Trials are different to ensure that convolution is performed along the time axis.
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
        which is the average of two consecutive values in both input trials
        (0 + 1)/2 = (0.25 + 0.75)/2 = 0.5.
    
    See Also
    --------
    numpy.tile
    """
    frates = np.tile(np.array([[0, 0.25], [1, 0.75]]), (5, 1)) # shape: (10, 2)
    tbin = 0.1
    window = 0.2
    mode = 'valid'
    expected = np.full(shape=(9, 2), fill_value=0.5)
    smoothed = smooth(frates, window, tbin, mode)
    assert smoothed.shape == expected.shape, "Wrong shape"
    assert_array_eq(smoothed, expected), "Wrong values"
