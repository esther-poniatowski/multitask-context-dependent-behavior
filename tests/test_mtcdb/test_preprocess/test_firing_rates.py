#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_preprocess.test_firing_rates` [module]

Tests for the module :mod:`mtcdb.preprocess.firing_rates`.

See Also
--------
np.testing.assert_array_almost_equal: 
    Assess element-wise equality between 2 arrays within a tolerance.
    Used to ignore numerical errors due to floating point arithmetic.
    Specifically designed for testing purposes (detailed error messages).
    Default tolerance: 1e-5.
"""

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_almost_equal as assert_array_eq
import pytest

from mtcdb.preprocess.firing_rates import extract_trial
from mtcdb.preprocess.firing_rates import slice_epoch
from mtcdb.preprocess.firing_rates import join_epochs
from mtcdb.preprocess.firing_rates import spikes_to_rates
from mtcdb.preprocess.firing_rates import smooth


spikes: npt.NDArray[np.float64] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
trials: npt.NDArray[np.int64] = np.array([1, 1, 2, 2, 3, 3])
expected: npt.NDArray[np.float64] = np.array([0.3, 0.4])

@pytest.mark.parametrize("spikes, trials, expected", 
                         argvalues = [
                             (spikes, trials, expected),
                             (np.array([]), np.array([]), np.array([]))
                             ],
                         ids = ["with_content", "emtpy"])
def test_extract_trial(spikes, trials, expected):
    """
    Tests for :func:`mtcdb.preprocess.firing_rates.extract_trial`.

    Parametrized for the argument ``data``.
    
    Test Inputs
    -----------
    trials [with_content]: :obj:`mtcdb.types.NumpyArray` of int
        3 trials.
    spikes [with_content]: :obj:`mtcdb.types.NumpyArray` of float
        2 spikes per trial.
    trials, spikes [empty]: :obj:`mtcdb.types.NumpyArray`
        No spikes (empty array). 
    trial [empty]: int
        Set to 2, to extract the spiking times of the second trial.
    """
    spk = extract_trial(trial=2, spikes=spikes, trials=trials)
    assert isinstance(spk, npt.NDArray), "Wrong type"
    assert spk.shape == expected.shape, "Wrong shape"
    assert_array_eq(spk, expected), "Wrong values"


spk = np.arange(0, 1, 0.1)
expected = np.array([0.0, 0.1, 0.2])
tstart, tend = 0.2, 0.5

@pytest.mark.parametrize("spk, expected", 
                         argvalues = [
                            (spk, expected),
                            (np.array([]), np.array([])) 
                            ],
                         ids = ["with_content", "empty"])
def test_slice_epoch(spk, expected):
    """
    Test for :func:`mtcdb.preprocess.firing_rates.slice_epoch`.

    Parametrized for the argument ``spk``.
    
    Test Inputs
    -----------
    spk [with_content] : :obj:`mtcdb.types.NumpyArray`
        Spikes evenly distributed in ``[0, 1]`` every 0.1 s (10 spikes).
        ``[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``
    spk [empty] : :obj:`mtcdb.types.NumpyArray`
        No spikes (empty array).
    tstart, tend: float
        Set to 0.2 and 0.5 as the start and end time of the epoch.
    
    Expected Outputs
    ----------------
    expected [with_content]: :obj:`mtcdb.types.NumpyArray`
        ``[0.0, 0.1, 0.2]``
        3 spikes are retained in the epoch, originally at ``[0.2, 0.3, 0.4]`` s 
        (since the starting time is included but not the ending time).
        Then, spiking times are shifted by the start time of the epoch,
        which requires to subtract 0.2 to each value.
    """
    sliced = slice_epoch(tstart, tend, spk)
    assert sliced.shape == expected.shape, "Wrong shape"
    assert_array_eq(sliced, expected), "Wrong values"


spk_0to2 = np.arange(0, 2, 0.1)
expected_0to2 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
spk_1to2 = np.arange(1, 2, 0.1)
expected_1to2 = np.array([0.4, 0.5, 0.6, 0.7])
tstart1, tend1 = 0.3, 0.7
tstart2, tend2 = 1.3, 1.7

@pytest.mark.parametrize("spk, expected", 
                         argvalues = [
                             (spk_0to2, expected_0to2),
                             (spk_1to2, expected_1to2),
                             (np.array([]), np.array([]))
                             ],
                         ids = ["0to2", "1to2", "empty"])
def test_join_epochs(spk, expected):
    """
    Test for :func:`mtcdb.preprocess.firing_rates.join_epochs`.

    Parametrized for the argument ``spk``.
    
    Test Inputs
    -----------
    spk [0to2]: :obj:`mtcdb.types.NumpyArray`
        Spikes evenly distributed in ``[0, 2]`` every 0.1 s (20 spikes).
    spk [1to2]: :obj:`mtcdb.types.NumpyArray`
        Spikes only after 1 s, so that there is no spike in the first epoch.
    spk [empty]: :obj:`mtcdb.types.NumpyArray`    
        No spikes at all (empty array).
    tstart1, tend1, tstart2, tend2: float
        Two epochs of 0.4 s duration : [0.3, 0.7] and [1.3, 1.7].

    Expected Outputs
    ----------------
    expected [0to2]: :obj:`mtcdb.types.NumpyArray`
        ``[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]``
        3 spikes are retained in each epoch, originally at
        ``[0.3, 0.4, 0.5, 0.6]`` and ``[1.3, 1.4, 1.5, 1.6]``
        Then, the spiking times of both epoch are shifted by 0.3 and 1.3 respectively:
        ``[0.0, 0.1, 0.2, 0.3]`` and ``[0.0, 0.1, 0.2, 0.3]``
        Finally, the second epoch is shifted  the duration of the first epoch
        (0.4 s) to be contiguous with the first epoch, hence the result.
    expected [1to2]: :obj:`mtcdb.types.NumpyArray`
        Spikes in the second epoch should still be retained
        and shifted at ``[0.4, 0.5, 0.6, 0.7]``.
    expected [empty]: :obj:`mtcdb.types.NumpyArray`
        No spike should be retained.
    """
    spk_joined = join_epochs(tstart1, tend1, tstart2, tend2, spk)
    assert spk_joined.shape == expected.shape, "Wrong shape"
    assert_array_eq(spk_joined, expected), "Wrong values"


spk = np.linspace(0, 1.0, num=20)
expected = np.full((10, 1), 20.0)
expected_empty = np.full((10, 1), 0.0)
t_max = 1.0
tbin = 0.1

@pytest.mark.parametrize("spk, expected", 
                          argvalues = [
                             (spk, expected),
                             (np.array([]), expected_empty)
                             ],
                         ids = ["with_content", "empty"])
def test_spikes_to_rates(spk, expected):
    """
    Test for :func:`mtcdb.preprocess.firing_rates.spikes_to_rates`.

    Parametrized for the argument ``spk``.
    
    Test Inputs
    -----------
    spk [with_content]: :obj:`mtcdb.types.NumpyArray`
        20 spikes evenly distributed in the recoding period.
    spk [empty]: :obj:`mtcdb.types.NumpyArray`
        No spikes at all (empty array).
    t_max: float
        Set to 1.0, for a recording period spanning ``[0, 1]``.
    tbin: float
        Set to 0.1, to divide the recording period into 10 time bins.
    
    Expected Outputs
    ----------------
    expected [with_content]: :obj:`mtcdb.types.NumpyArray`
        10 values (10 time bins) of 20 spikes/s.
        Shape: ``(10, 1)`` (10 time bins, 1 trial).
    expected [empty]: :obj:`mtcdb.types.NumpyArray`
        Idem with 0 spikes/s.
	"""
    frates = spikes_to_rates(spk, tbin, t_max)
    assert frates.shape == expected.shape, "Wrong shape"
    assert_array_eq(frates, expected), "Wrong values"


frates = np.tile(np.array([[0, 0.25], [1, 0.75]]), (5, 1)) # shape: (10, 2)
tbin = 0.1
window = 0.2
mode = 'valid'
expected = np.full(shape=(9, 2), fill_value=0.5)

def test_smooth():
    """
    Test for :func:`mtcdb.preprocess.firing_rates.smooth`.

    Test Inputs
    -----------
    frates: :obj:`mtcdb.types.NumpyArray`
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
    expected: :obj:`mtcdb.types.NumpyArray`
        Shape: ``(7, 2)`` (7 time bins, 2 trials).
        In the ``'valid'`` mode, the output time dimension is given by:
        ``ntpts_out = ntpts - window/tbin + 1``.
        Here: ``window/tbin = 2`` bins, so ``ntpts_out = 10 - 2 + 1 = 9``.
        Output values: All equal to 0.5, 
        which is the average of two consecutive values in both input trials
        (0 + 1)/2 = (0.25 + 0.75)/2 = 0.5.
    
    Notes
    -----
    No need to parametrize this test for an empty input array: 
    the firing rate time courses will be of fixed duration, 
    hence non-empty (even if all values are 0.0).

    See Also
    --------
    numpy.tile
    """
    smoothed = smooth(frates, window, tbin, mode)
    assert smoothed.shape == expected.shape, "Wrong shape"
    assert_array_eq(smoothed, expected), "Wrong values"

