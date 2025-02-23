#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.convert_to_rates` [module]

Classes
-------
FiringRatesConverter
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `process` method in `FiringRatesConverter`.
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------

from typing import TypeAlias, Any, Tuple, Optional

import numpy as np
from scipy.signal import fftconvolve

from core.constants import T_BIN, T_MAX
from core.processors.base_processor import Processor


SpikingTimes: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for spiking times."""

FiringRates: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.float64]]
"""Type alias for firing rates."""


class FiringRatesConverter(Processor):
    """
    Convert raw spike times into firing rates.

    Attributes
    ----------
    t_bin : float
        Time bin (in seconds).
    t_max : float
        Duration of the recording period (in seconds).
    smooth_window : float
        Size of the smoothing window (in seconds).
    mode : str
        Convolution mode for smoothing. Options: ``'valid'`` (default), ``'same'``.
        See the `smooth` method.
    n_t : int
        (Property). Number of time bins in the firing rate time course ``f_binned``.
    n_t_smth : int
        (Property) Number of time bins in the smoothed firing rate time course ``f_smoothed``,
        depending on the convolution mode. See the `smooth` method.

    Methods
    -------
    spikes_to_rates
    smooth

    Examples
    --------
    Consider spiking times recorded during in a period of 1 second duration, homogeneously
    distributed every 0.1 s:

    >>> spikes = np.arange(0, 1, 0.1)
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    The firing rate is expected to be 10 Hz:

    >>> converter = FiringRatesConverter(t_bin=0.1, t_max=1.0, smooth_window=0.5)
    >>> f_rates = converter.process(spikes)
    >>> print(f_rates)
    [0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
    """

    def __init__(
        self,
        t_bin: float = T_BIN,
        t_max: float = T_MAX,
        smooth_window: float = 0.5,
        mode: str = "valid",
    ):
        self.t_bin = t_bin
        self.t_max = t_max
        self.smooth_window = smooth_window
        self.mode = mode

    @property
    def n_t(self) -> int:
        """
        Number of time bins in the recording period.

        Rule: ``n_t = t_max / t_bin``. If the division is not exact, the result is rounded down.
        """
        return int(self.t_max / self.t_bin)

    @property
    def n_t_smth(self) -> int:
        """
        Number of time bins in the smoothed firing rate time course.

        Rule: Depend on the convolution mode. See :meth:`smooth` for details.
        """
        if self.mode == "same":
            return self.n_t
        elif self.mode == "valid":
            return self.n_t - int(self.smooth_window / self.t_bin) + 1
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def process(self, spikes: SpikingTimes) -> FiringRates:
        """
        Implement the abstract method of the base class `Processor`.

        Arguments
        ---------
        spikes: SpikingTimes
            Spiking times in seconds, relative to time ``t=0``.
            Shape: ``(n_spikes,)``
            .. _spikes:

        Returns
        -------
        f_rates: FiringRates
            Firing rate time course in spikes/s.
            Shape: ``(n_t_smth,)`` (see attribute `n_t_smth`).

        Notes
        -----
        The conversion to firing rates consists of two steps:

        1. Binning the spikes.
        2. Smoothing the binned rates.
        """
        f_binned = self.spikes_to_rates(spikes, self.t_max, self.t_bin)
        f_smooth = self.smooth(f_binned, self.t_bin, self.smooth_window, self.mode)
        return f_smooth

    @staticmethod
    def spikes_to_rates(spikes: SpikingTimes, t_max: float, t_bin: float) -> FiringRates:
        """
        Convert a spikes train into a firing rate time course.

        Arguments
        ---------
        spikes : SpikingTimes
            See the argument :ref:`spikes` in the `process` method.
        t_max : float
            Duration of the recording period (in seconds).
        t_bin : float
            Time bin of the firing rate time course (in seconds).

        Returns
        -------
        f_binned : FiringRates
            Firing rate time course in spikes/s, obtained by binning the spikes.
            Shape: ``(n_t,)``, with ``n_t = t_max / t_bin`` the number of time bins.
            .. _f_binned:

        Implementation
        --------------
        - Divide the recording period ``[0, t_max]`` into bins of size ``t_bin``.
        - Count the number of spikes in each bin.
        - Divide the spikes count in each bin by the bin size ``t_bin`` to get rates in spikes/s.

        Bin edges span the duration from 0 to ``t_max`` by steps of ``t_bin``.
        With :func:`numpy.arange`, if the step falls at ``t_max``, then the last bin edge be
        excluded from the sequence. To include it in any case, the stop value is ``t_max + t_bin``.

        See Also
        --------
        :func:`numpy.histogram(arr, bins)`
            Used to count the number of spikes in each bin. Input `bins` contains the starting time
            of each bin, so the last bin edge should be included in this sequence. Outputs: ``hist``
            (number of spikes in each bin), ``edges`` (useless here).
        """
        hist, _ = np.histogram(spikes, bins=np.arange(0, t_max + t_bin, t_bin))
        f_binned = hist / t_bin
        return f_binned

    @staticmethod
    def smooth(f_binned: FiringRates, t_bin: float, smooth_window: float, mode: str) -> FiringRates:
        """
        Smooth the firing rates across time.

        Arguments
        ---------
        f_binned : FiringRates
            See the return value :ref:`f_binned`.
        t_bin : float
            Time bin of the firing rate time course (in seconds).
        smooth_window : float
            Size of the smoothing window (in seconds).
        mode : str
            Convolution mode for smoothing. Options: ``'valid'`` (default), ``'same'``.
            See the Notes section.

        Returns
        -------
        f_smoothed : FiringRates
            Smoothed firing rate time course in spikes/s.
            Shape: ``(n_t_smth,)``, with ``n_t_smth`` the number of time bins in the smoothed time
            course. The number of time bins depends on the convolution mode.
            .. _f_smoothed:

        Notes
        -----
        Convolution Modes:

        - ``'same'``:
            - Keep the same output shape as the input sequence. The kernel is centered on each input
              element, with zero-padding applied to the edges of the input signal as needed.
            - Output shape: ``n_out = n_in``. Here: ``n_t_smth = n_t``.
        - ``'valid'``:
            - Keep only the values which are not influenced by zero-padding, i.e. where the kernel
              fully overlaps with the input signal. The first valid position is obtained when the
              kernel's left edge aligns with the input's left edge. The last valid position is
              obtained when the kernel's right edge aligns with the input's right edge.
            - Output shape: ``n_out = n_in - k + 1``, where ``k`` is the kernel size. Indeed, `k -
              1` positions have to be subtracted since the kernel cannot fit within the signal when
              it is placed on the last `k + 1` positions.
              Here: ``n_t_smth = n_t - smooth_window/t_bin + 1``.

        Implementation
        --------------
        Smoothing consists in averaging consecutive values in a sliding smooth_window.

        - Define a boxcar kernel with all values equal to 1. The window size (in number of bins) is
          equal to ``smooth_window/t_bin`` (rounded down to the nearest integer) to match the time
          bin of the firing rate time course.
        - Convolve the firing rate time course with the boxcar kernel (FFT method), to *sum* the
          values in the window at each location in the time course.
        - Divide the output by the window size to get the *average*. This is necessary to keep the
          same scale as the input firing rates, and to avoid increasing the values when the
          smooth_window size is large.

        See Also
        --------
        :func:`scipy.signal.fftconvolve(arr, kernel, mode, axes)`
            Convolve the firing rate time course with kernel.
        """
        kernel = np.ones(int(smooth_window / t_bin))  # boxcar kernel
        f_smoothed = fftconvolve(f_binned, kernel, mode=mode, axes=0) / len(kernel)
        return f_smoothed
