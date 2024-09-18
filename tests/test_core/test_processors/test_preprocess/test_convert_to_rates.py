#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_pipelines.test_preprocess.test_convert_to_rates` [module]

Tests for the module :mod:`core.processors.preprocess.convert_to_rates`.

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

from processors.preprocess.convert_to_rates import FiringRatesConverter


def test_spikes_to_rates():
    """
    Test for :meth:`spikes_to_rates`.

    Test Inputs
    -----------
    spk [with_content]: :obj:`core.types.NumpyArray`
        20 spikes evenly distributed in the recoding period ``[0, 1]``.
    t_max: float
        Set to 1.0, sec, to include the full recording period.
    t_bin: float
        Set to 0.1 sec, to divide the recording period into 10 time bins.

    Expected Outputs
    ----------------
    expected: np.ndarray
        10 values (10 time bins) of 20 spikes/s.
        Shape: ``(10, 1)`` (10 time bins, 1 trial).
    """
    spikes = np.linspace(0, 1.0, num=20)
    expected = np.full(shape=10, fill_value=20.0)
    t_max = 1.0
    t_bin = 0.1
    converter = FiringRatesConverter(t_bin=t_bin, t_max=t_max)
    converter.spikes = spikes  # set manually
    converter.spikes_to_rates()
    f_binned = converter.f_binned
    assert f_binned.shape == expected.shape, "Wrong shape"
    assert_array_eq(f_binned, expected), "Wrong values"


def test_smooth():
    """
    Test for :meth:`smooth`.

    Test Inputs
    -----------
    f_binned: np.ndarray
        Alternating patterns [0, 1], repeated 5 times (10 values).
    t_bin: float
        Set to 0.1 sec.
    smooth_window: int
        Set to 0.2, for a boxcar kernel of 2 bins width.
    mode: str
        Set to ``'valid'``, to keep only the values which are not influenced by zero-padding.

    Expected Outputs
    ----------------
    expected: np.ndarray
        Shape: ``(n_tpts_smth,)`` with ``n_tpts_smth = n_tpts - smooth_window/t_bin + 1`` (valid mode).
        Here: ``smooth_window/t_bin = 2`` bins, so ``n_tpts_smth = 10 - 2 + 1 = 9``.
        Values: All equal to 0.5, which is the average of two consecutive values in both input
        trials, ``(0 + 1)/2 = (0.25 + 0.75)/2 = 0.5``.

    Notes
    -----
    This input is not usual since its values are not monotonically increasing like in a time series
    in the experiment. It is used for simplicity to test the smoothing operation.

    See Also
    --------
    :func:`numpy.tile`: Repeat the input array along a specified axis.
    """
    f_binned = np.tile(np.array([0, 1]), 5)  # shape: (10,)
    t_bin = 0.1
    t_max = 1.0  # not used by the `smooth` method, only to initialize the converter
    smooth_window = 0.2
    mode = "valid"
    expected = np.full(shape=9, fill_value=0.5)
    converter = FiringRatesConverter(
        t_bin=t_bin, t_max=t_max, smooth_window=smooth_window, mode=mode
    )
    converter.f_binned = f_binned  # set manually
    converter.smooth()
    smoothed = converter.f_smoothed
    assert smoothed.shape == expected.shape, f"Output shape: {smoothed.shape} != {expected.shape}"
    assert_array_eq(smoothed, expected), f"Output values: {smoothed} != {expected}"
