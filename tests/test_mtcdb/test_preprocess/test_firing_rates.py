#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_preprocess.test_firing_rates` [module]
============================================================

Tests for the module :mod:`mtcdb.preprocess.firing_rates`.
"""

import numpy as np
import pytest

from mtcdb.preprocess.firing_rates import spikes_to_rates

def test_bin_spikes():
    """Test for :func:`mtcdb.preprocess.firing_rates.spikes_to_rates`.
    
    Test Inputs
    -----------
    tmax: float
        Set to 1.0, for a recording period spanning [0, 1].
    tbin: float
        Set to 0.1, to divide the recording period into 10 bins.
    spk: numpy.ndarray
        20 spikes evenly distributed in the recoding period.
    
    Expected Outputs
    ----------------
    expected: numpy.ndarray
        10 values (10 bins) of 20 spikes/s.
	
    See Also
    --------
    numpy.allclose: 
        Assess element-wise equality between 2 arrays within a tolerance.
        Used to ignore numerical errors due to floating point arithmetic.
	"""
    tmax = 1.0
    tbin = 0.1
    spk = np.linspace(0, tmax, num=20)
    expected = np.full(shape=10, fill_value=20.0)
    frates = spikes_to_rates(spk, tbin, tmax)
    assert frates.shape == expected.shape, "Wrong shape"
    assert np.allclose(frates, expected), "Wrong values"

