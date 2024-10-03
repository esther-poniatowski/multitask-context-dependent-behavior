#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.convert_to_rates` [module]

Classes
-------
`FiringRatesConverter`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member

from typing import TypeAlias, Any, Tuple, Optional

import numpy as np
from scipy.signal import fftconvolve

from core.constants import T_BIN
from core.processors.base_processor import Processor


SpikingTimes: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for spiking times."""

FiringRates: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.float64]]
"""Type alias for firing rates."""


class FiringRatesConverter(Processor):
    """
    Convert raw spike times into firing rates.

    Conventions for the documentation:

    - Attributes: Configuration parameters of the processor, passed to the *constructor*.
    - Arguments: Input data to process, passed to the `process` method (base class).
    - Returns: Output data after processing, returned by the `process` method (base class).

    Attributes
    ----------
    t_bin: float
        Time bin (in seconds).
    t_max: float
        Duration of the recording period (in seconds).
    smooth_window: float
        Size of the smoothing window (in seconds).
    mode: str
        Convolution mode for smoothing. Options: ``'valid'`` (default), ``'same'``.
        See method `smooth` for details.
    n_t: int
        Number of time bins in the firing rate time course ``f_binned``: ``n_t = t_max/t_bin``.
    n_t_smth: int
        Number of time bins in the smoothed firing rate time course ``f_smoothed``, depending on the
        convolution mode. See method `smooth` for details.

    Arguments
    ---------
    spikes: SpikingTimes
        Spiking times. Shape: ``(n_spikes,)``
        .. _spikes:

    Returns
    -------
    f_rates: FiringRates
        Firing rate time course (in spikes/s), obtained in two steps:
        1. Binning the spikes.
        2. Smoothing the binned rates.
        Shape: ``(n_t_smth,)`` (see attribute `n_t_smth`).
        .. _f_rates:

    Methods
    -------
    spikes_to_rates
    smooth

    Examples
    --------
    Consider spiking times recorded during in two periods of 1 second duration, first homogeneously
    distributed every 0.1 s, and then every 0.2 s. The firing rate is expected to be 10 Hz in the
    first period and 5 Hz in the second period.

    >>> spikes_1 = np.arange(0, 1, 0.1)
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    >>> spikes_2 = np.arange(0, 1, 0.2)
    [0.0, 0.2, 0.4, 0.6, 0.8]
    >>> converter = FiringRatesConverter(t_bin=0.1, t_max=1.0, smooth_window=0.5)
    >>> f_rates = converter.process(spikes=spikes1)
    >>> print(f_rates)
    [0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    >>> f_rates_2 = converter.process(spikes=spikes_2)
    >>> print(f_rates_2)
    [0. 0. 1. 0. 1. 0. 1. 0. 1. 0.]

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random = False

    def __init__(
        self,
        t_bin: float = T_BIN,
        t_max: Optional[float] = None,
        smooth_window: float = 0.5,
        mode: str = "valid",
    ):
        super().__init__(t_bin=t_bin, t_max=t_max, smooth_window=smooth_window, mode=mode)

    def _process(self, **input_data: Any) -> FiringRates:
        """Implement the template method called in the base class `process` method."""
        spikes = input_data["spikes"]
        f_binned = self.spikes_to_rates(spikes)
        f_smooth = self.smooth(f_binned)
        return f_smooth

    def spikes_to_rates(self, spikes: SpikingTimes) -> FiringRates:
        """
        Convert a spike train into a firing rate time course.

        Arguments
        ---------
        spikes: SpikingTimes
            See the argument :ref:`spikes`.

        Returns
        -------
        f_binned: FiringRates
            Firing rate time course (in spikes/s), obtained by binning the spikes.
            Shape: ``(n_t,)`` (see :attr:`FiringRatesConverter.n_t`).
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
        hist, _ = np.histogram(spikes, bins=np.arange(0, self.t_max + self.t_bin, self.t_bin))
        f_binned = hist / self.t_bin
        return f_binned

    def smooth(self, f_binned: FiringRates) -> FiringRates:
        """
        Smooth the firing rates across time.

        Arguments
        ---------
        f_binned: FiringRates
            See the return value :ref:`f_binned`.

        Returns
        -------
        f_smoothed: FiringRates
            Smoothed firing rate time course (in spikes/s).
            Shape: ``(n_t_smth,)`` (see the attribute `n_t_smth`).
            .. _f_smoothed:

        Notes
        -----
        Convolution Modes:

        - ``'same'``: Keep the same output shape as the input sequence. The kernel is centered on
          each input element, with zero-padding applied to the edges of the input signal as needed.
        - ``'valid'``: Keep only the values which are not influenced by zero-padding, i.e. where the
          kernel fully overlaps with the input signal. The first valid position is when the kernel's
          left edge aligns with the input's left edge. The last valid position is when the kernel's
          right edge aligns with the input's right edge.

        Corresponding output shape:

        - ``'same'``:  ``n_out = n_in``. Here: ``n_t_smth = n_t``.
        - ``'valid'``:  ``n_out = n_in - k + 1``, where ``k`` is the kernel size. Indeed, `k - 1`
          positions have to be subtracted since the kernel cannot fit within the signal when it is
          placed on the last `k + 1` positions. Here: ``n_t_smth = n_t - smooth_window/t_bin + 1``.

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
        kernel = np.ones(int(self.smooth_window / self.t_bin))  # boxcar kernel
        f_smoothed = fftconvolve(f_binned, kernel, mode=self.mode, axes=0) / len(kernel)
        return f_smoothed

    @property
    def n_t(self) -> int:
        """
        Number of time bins in the recording period.

        Rule: ``n_t = t_max/t_bin``. If the division is not exact, the result is rounded down.
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
