"""
:mod:`mtcdb.preprocess.firing_rates` [module]
=============================================

Convert raw spike times to firing rates.

* SPIKES (dict) 
    Spiking times of all units in all sessions. Nested dictionary.
        Keys 1 : Units.
        Keys 2 : Sessions.
        Values : Spiking times in all the trials of the session 
        (array of the form [[trial numbers], [spike indices]])
"""

import numpy as np
from scipy.signal import fftconvolve
from typing import Any

from mtcdb.constants import TBIN
from mtcdb.types import ArrayLike, NumpyArray


def extract_trial(trial:int, data:NumpyArray) -> NumpyArray:
    """
    Extract the spiking times in one specific trial.

    Parameters
    ----------
    trial: int
        Number of the trial of interest.
    data: NumpyArray
        Raw data corresponding to a *whole session*, for one unit.
        Shape: ``(2, nspikes)``.
        ``data[1]``: Spiking times in seconds (starting from 0 in each trial).
        ``data[0]``: Trial in which each spike occurred.
    
    Returns
    -------
    spk: NumpyArray
        Spiking times occurring in the selected trial.
        Shape: ``(nspikes_trial,)``.
    """
    return data[1][data[0]==trial]


def slice_epoch(t0: float, t1: float, spk: NumpyArray) -> NumpyArray:
    """
    Extract spiking times within one epoch of one trial.

    Important
    ---------
    Spiking times are *relative* to the start of the epoch.

    Parameters
    ----------
    t0: float
        Start time of the epoch (in seconds).
    t1: float
        End time of the epoch (in seconds).
    spk: NumpyArray
        Spiking times during a *whole trial* (in seconds).
        Shape: ``(nspikes,)``.
    
    Returns
    -------
    spk_times_epoch: NumpyArray
        Spiking times with the epoch comprised between t0 and t1,
        relative to the start of the epoch.
        Shape: ``(nspikes_epoch, 1)``.
    
    Implementation
    --------------
    - Select the spiking times within the epoch with a boolean mask.
    - Subtract the starting time of the epoch to reset the time.
    """
    return spk[(spk>=t0)&(spk<t1)] - t0



def align_spikes(spikes: Any) -> Any:
    pass


def spikes_to_rates(spk: ArrayLike,
                    tbin: float,
                    tmax: float,
                    ) -> NumpyArray:
    """
    Convert a spike train into a firing rate time course.
    
    Parameters
    ----------
    spk: :obj:`mtcdb.types.ArrayLike`
        Spiking times.
    tbin: float
        Time bin (in seconds).
    tmax: float
        Duration of the recording period (in seconds).
    
    Returns
    -------
    frates: :obj:`mtcdb.types.NumpyArray`
        Firing rate time course (in spikes/s).
        Shape: ``(ntpts, 1)`` with ``ntpts = tmax/tbin`` (number of bins).
    
    See Also
    --------
    numpy.histogram: Used to count the number of spikes in each bin.
    
    Algorithm
    ---------

    - Divide the recording period  ``[0, tmax]`` into bins of size ``tbin``.
    - Count the number of spikes in each bin.
    - Divide the spikes count in each bin by the bin size ``tbin``.

    Implementation
    --------------

    :func:`np.histogram` takes an argument `bins` for bin edges,
    which should include the *rightmost edge*.
    Bin edges are obtained with :func:`numpy.arange`, 
    with the last bin edge at ``tmax + tbin`` to include the last bin.
    :func:`np.histogram` returns two outputs: 
    ``hist`` (number of spikes in each bin), ``edges`` (useless).
    
    The shape of ``frates`` is extended to two dimensions representing
    time (length ``n_bins``),
    trials (length ``1``, single trial).
    It ensures compatibility and consistence in the full process.
    """
    frates = np.histogram(spk, bins=np.arange(0, tmax+tbin, tbin))[0]/tbin
    frates = frates[:,np.newaxis] # add one dimension for trials
    return frates


def smooth(frates: NumpyArray,
           window: float,
           tbin: float,
           mode: str = 'valid',
           ) -> NumpyArray:
    """
    Smooth the firing rates across time.

    Parameters
    ----------
    frates: :obj:`mtcdb.types.NumpyArray`
        Firing rate time course (in spikes/s).
        Shape: ``(ntpts, ntrials)``,
    window: float
        Smoothing window size (in seconds).
    tbin: float
        Time bin (in seconds).
    
    Returns
    -------
    smoothed: :obj:`mtcdb.types.NumpyArray`
        Smoothed firing rate time course (in spikes/s).
        Shape: ``(ntpts_out, ntrials)``, ``ntpts_out`` depend on ``mode``.
        With ``"valid"``:  ``ntpts_out = ntpts - window/tbin + 1``.
        With ``"same"``:  ``ntpts_out = ntpts``.
    
    See Also
    --------
    scipy.signal.fftconvolve: Used to convolve the firing rate time course with a boxcar kernel.
    
    Notes
    -----
    Smoothing consists in averaging consecutive values in a sliding window.

    Algorithm

    - Convolve the firing rate time course with a boxcar kernel (FFT method).
      Size of the window: ``window/tbin``.
    - Divide the output by the window size to get the average.

    Convolution Modes

    - ``'same'``: Keep the output shape as the input sequence.
    - ``'valid'``: Keep only the values which are not influenced by zero-padding.
    """
    kernel = np.ones((int(window/tbin), 1)) # add one dimension for shape compatibility
    smoothed = fftconvolve(frates, kernel, mode=mode, axes=0)/len(kernel)
    return smoothed
    
