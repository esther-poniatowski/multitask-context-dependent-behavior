#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.firing_rates` [module]

Classes
-------
:class:`FiringRatesConvertor`
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve

from core.constants import T_BIN


SpikingTimes: TypeAlias = npt.NDArray[np.float64]
"""Type alias for spiking times."""
FiringRates: TypeAlias = npt.NDArray[np.float64]
"""Type alias for firing rates."""


class FiringRatesConvertor:
    """
    Convert raw spike times into firing rates.

    Attributes
    ----------

    Methods
    -------
    extract_trial
    slice_epoch
    join_epochs
    align_timings
    spikes_to_rates
    smooth
    """

    def __init__(self, spikes: SpikingTimes, trials: npt.NDArray[np.int64]):
        self.spikes = spikes
        self.trials = trials

    @staticmethod
    def spikes_to_rates(spk: SpikingTimes, t_bin: float, t_max: float) -> FiringRates:
        """
        Convert a spike train into a firing rate time course.

        Parameters
        ----------
        spk: :obj:`core.types.ArrayLike`
            Spiking times.
        t_bin: float
            Time bin (in seconds).
        t_max: float
            Duration of the recording period (in seconds).

        Returns
        -------
        frates: :obj:`core.types.NumpyArray`
            Firing rate time course (in spikes/s).
            Shape: ``(ntpts, 1)`` with ``ntpts = t_max/t_bin`` (number of bins).

        See Also
        --------
        numpy.histogram: Used to count the number of spikes in each bin.

        Algorithm
        ---------
        - Divide the recording period ``[0, t_max]`` into bins of size ``t_bin``.
        - Count the number of spikes in each bin.
        - Divide the spikes count in each bin by the bin size ``t_bin``.

        Implementation
        --------------
        :func:`np.histogram` takes an argument `bins` for bin edges,
        which should include the *rightmost edge*.
        Bin edges are obtained with :func:`numpy.arange`,
        with the last bin edge at ``t_max + t_bin`` to include the last bin.
        :func:`np.histogram` returns two outputs:
        ``hist`` (number of spikes in each bin), ``edges`` (useless).

        The shape of ``frates`` is extended to two dimensions representing
        time (length ``n_bins``),
        trials (length ``1``, single trial).
        It ensures compatibility and consistence in the full process.
        """
        frates = np.histogram(spk, bins=np.arange(0, t_max + t_bin, t_bin))[0] / t_bin
        frates = frates[:, np.newaxis]  # add one dimension for trials
        return frates

    @staticmethod
    def smooth(
        frates: FiringRates, window: float, t_bin: float, mode: str = "valid"
    ) -> FiringRates:
        """
        Smooth the firing rates across time.

        Parameters
        ----------
        frates: :obj:`core.types.NumpyArray`
            Firing rate time course (in spikes/s).
            Shape: ``(ntpts, ntrials)``,
        window: float
            Smoothing window size (in seconds).
        t_bin: float
            Time bin (in seconds).

        Returns
        -------
        smoothed: :obj:`core.types.NumpyArray`
            Smoothed firing rate time course (in spikes/s).
            Shape: ``(ntpts_out, ntrials)``, ``ntpts_out`` depend on ``mode``.
            With ``"valid"``:  ``ntpts_out = ntpts - window/t_bin + 1``.
            With ``"same"``:  ``ntpts_out = ntpts``.

        See Also
        --------
        scipy.signal.fftconvolve: Used to convolve the firing rate time course with a boxcar kernel.

        Algorithm
        ---------
        Smoothing consists in averaging consecutive values in a sliding window.

        - Convolve the firing rate time course with a boxcar kernel (FFT method).
        Size of the window: ``window/t_bin``.
        - Divide the output by the window size to get the average.

        Convolution Modes

        - ``'same'``: Keep the output shape as the input sequence.
        - ``'valid'``: Keep only the values which are not influenced by zero-padding.
        """
        kernel = np.ones((int(window / t_bin), 1))  # add one dimension for shape compatibility
        smoothed = fftconvolve(frates, kernel, mode=mode, axes=0) / len(kernel)
        return smoothed

    def process_firing_rates(self, trial: int, t_bin: float, t_max: float) -> FiringRates:
        """
        Complete pipeline to process spiking data for a given trial into firing rates.
        """
        spk_trial = None  # self.extract_trial(trial)
        return self.spikes_to_rates(spk_trial, t_bin, t_max)
