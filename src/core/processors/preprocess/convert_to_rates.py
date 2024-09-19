#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.convert_to_rates` [module]

Classes
-------
:class:`FiringRatesConverter`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init

from types import MappingProxyType
from typing import TypeAlias, Any, Tuple, Optional

import numpy as np
from scipy.signal import fftconvolve

from core.constants import T_BIN
from processors.base_processor import Processor
from utils.misc.arrays import create_empty_array


SpikingTimes: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for spiking times."""

FiringRates: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.float64]]
"""Type alias for firing rates."""


class FiringRatesConverter(Processor):
    """
    Convert raw spike times into firing rates.

    Attributes
    ----------
    spikes: np.ndarray[Tuple[Any], np.dtype[np.int64]]
        Spiking times. Shape: ``(n_spikes,)``.
    f_binned: np.ndarray[Tuple[Any], np.dtype[np.float64]]
        Firing rate time course (in spikes/s), obtained by binning the spikes.
        Shape: ``(n_tpts,)`` (see :attr:`n_tpts`).
    f_smoothed: np.ndarray[Tuple[Any], np.dtype[np.float64]]
        Smoothed firing rate time course (in spikes/s).
        Shape: ``(n_tpts_smth,)`` (see :attr:`n_tpts_smth`).
    n_tpts: int
        Number of time bins in the recording period, corresponding to the final length of the binned
        firing rates time course ``f_binned``: ``n_tpts = t_max/t_bin``.
    n_tpts_smth: int
        Number of time bins in the smoothed firing rate time course, corresponding to the final
        length of the smoothed firing rates time course ``f_smoothed``.
    t_bin: float
        Time bin (in seconds).
    t_max: float
        Duration of the recording period (in seconds).
    smooth_window: float
        Size of the smoothing window (in seconds).
    mode: str
        Convolution mode for smoothing. Options: ``'valid'`` (default), ``'same'``. See
        :meth:`smooth` for details.

    Methods
    -------
    spikes_to_rates
    smooth

    Examples
    --------
    Assume spiking times homogeneously distributed every 0.1 s in a 1-second recording period, and
    then homogeneously distributed every 0.2 s in another 1-second recording period. The firing rate
    is expected to be 10 Hz in the first period and 5 Hz in the second period.

    >>> converter = FiringRatesConverter(t_bin=0.1, t_max=1.0, smooth_window=0.5)
    >>> spikes1 = np.arange(0, 1, 0.1)
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    >>> converter.process(spikes=spikes1)
    >>> print(converter.f_binned)
    [0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    >>> spikes2 = np.arange(0, 1, 0.2)
    [0.0, 0.2, 0.4, 0.6, 0.8]
    >>> converter.process(spikes=spikes2)
    >>> print(converter.f_binned)
    [0. 0. 1. 0. 1. 0. 1. 0. 1. 0.]

    Applying a 0.5 s smoothing window over the total firing rate time course will average the rates
    over the window:

    - For the first 0.25 seconds, only spikes from the first period (10 Hz) will contribute to the
      smoothed rate.
    - Between 0.25 s and 0.75 s, the smoothing window will include more spikes from the first
      period, and the rate will be dominated by the 10 Hz spikes.
    - After 1 second, the window starts to overlap both the first and second periods. The rate will
      gradually decrease as spikes from the second period (5 Hz) enter the window.

    >>> spikes_tot = np.concatenate((spikes1, spikes2 + 1))
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2, 1.4, 1.6, 1.8, 2.0]
    >>> converter.t_max = 2.0
    >>> converter.process(spikes=spikes_tot)
    >>> print(converter.f_smoothed)
    #TODO: Add expected output

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors. See definition of class-level attributes and template
        methods.
    """

    config_attrs = ("t_bin", "t_max", "smooth_window", "mode")
    input_attrs = ("spikes",)
    output_attrs = ("f_binned", "f_smoothed")
    empty_data = MappingProxyType(
        {
            "spikes": create_empty_array(1, np.int64),
            "f_binned": create_empty_array(1, np.float64),
            "f_smoothed": create_empty_array(1, np.float64),
        }
    )

    def __init__(
        self,
        t_bin: float = T_BIN,
        t_max: Optional[float] = None,
        smooth_window: float = 0.5,
        mode: str = "valid",
    ):
        super().__init__(t_bin=t_bin, t_max=t_max, smooth_window=smooth_window, mode=mode)

    def _process(self) -> None:
        """Implement the template method called in the base class :meth:`process` method."""
        self.spikes_to_rates()
        self.smooth()

    def spikes_to_rates(self) -> None:
        """
        Convert a spike train into a firing rate time course.

        Important
        ---------
        Update the attribute `f_binned` with the computed firing rates.

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
        hist, _ = np.histogram(self.spikes, bins=np.arange(0, self.t_max + self.t_bin, self.t_bin))
        f_binned = hist / self.t_bin
        self.f_binned = f_binned

    def smooth(self) -> None:
        """
        Smooth the firing rates across time.

        Important
        ---------
        Update the attribute `f_smooth` with the smoothed firing rates.

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

        - ``'same'``:  ``n_out = n_in``. Here: ``n_tpts_smth = n_tpts``.
        - ``'valid'``:  ``n_out = n_in - k + 1``, where ``k`` is the kernel size. This is because
          the kernel cannot fit in the signal when it is placed on the last `k+1` positions. Here:
          ``n_tpts_smth = n_tpts - smooth_window/t_bin + 1``.

        Implementation
        --------------
        Smoothing consists in averaging consecutive values in a sliding smooth_window.

        - Define a boxcar kernel with all values equal to 1. The window size (in number of bins) is
          equal to ``smooth_window/t_bin`` (rounded down to the nearest integer) to match the time bin of
          the firing rate time course.
        - Convolve the firing rate time course with the boxcar kernel (FFT method), to *sum* the
          values in the window at each location in the time course.
        - Divide the output by the window size to get the *average*. This is necessary to keep the
          same scale as the input firing rates, and to avoid increasing the values when the smooth_window
          size is large.

        See Also
        --------
        :func:`scipy.signal.fftconvolve(arr, kernel, mode, axes)`
            Convolve the firing rate time course with kernel.
        """
        kernel = np.ones(int(self.smooth_window / self.t_bin))  # boxcar kernel
        f_smoothed = fftconvolve(self.f_binned, kernel, mode=self.mode, axes=0) / len(kernel)
        self.f_smoothed = f_smoothed

    @property
    def n_tpts(self) -> int:
        """
        Number of time bins in the recording period.

        Rule: ``n_tpts = t_max/t_bin``. If the division is not exact, the result is rounded down.
        """
        return int(self.t_max / self.t_bin)

    @property
    def n_tpts_smth(self) -> int:
        """
        Number of time bins in the smoothed firing rate time course.

        Rule: Depend on the convolution mode. See :meth:`smooth` for details.
        """
        if self.mode == "same":
            return self.n_tpts
        elif self.mode == "valid":
            return self.n_tpts - int(self.smooth_window / self.t_bin) + 1
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
