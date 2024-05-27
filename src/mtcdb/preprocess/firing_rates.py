"""
:mod:`mtcdb.preprocess.firing_rates` [module]

Convert raw spike times to firing rates.

See Also
--------
test_mtcdb.test_preprocess.test_firing_rates:
    Unit tests for this module.
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
    data: :obj:`mtcdb.types.NumpyArray`
        Raw data corresponding to a *whole session*, for one unit.
        Shape: ``(2, nspikes)`` (see Implementation section).
    
    Returns
    -------
    spk: :obj:`mtcdb.types.NumpyArray`
        Spiking times occurring in the selected trial.
        Shape: ``(nspikes_trial,)``.
    
    Implementation
    --------------
    ``data[1]``: Spiking times in seconds (starting from 0 in each trial).
    ``data[0]``: Trial in which each spike occurred.
    To extract the spiking times of one trial, use a boolean mask on the trial number.
    """
    return data[1][data[0]==trial]


def slice_epoch(tstart: float, tend: float, 
                spk: NumpyArray) -> NumpyArray:
    """
    Extract spiking times within one epoch of one trial.

    Important
    ---------
    Spiking times are *relative* to the beginning of the epoch.

    Parameters
    ----------
    tstart, tend: float
        Times boundaries of the epoch (in seconds).
    spk: :obj:`mtcdb.types.NumpyArray`
        Spiking times during a *whole trial* (in seconds).
        Shape: ``(nspikes,)``.
    
    Returns
    -------
    spk_epoch: :obj:`mtcdb.types.NumpyArray`
        Spiking times in the epoch comprised between ``tstart`` and ``tend``,
        reset to be relative to the beginning of the epoch.
        Shape: ``(nspikes_epoch, 1)``.
    
    Implementation
    --------------
    - Select the spiking times within the epoch with a boolean mask.
    - Subtract the starting time of the epoch to reset the time.
    """
    return spk[(spk>=tstart)&(spk<tend)] - tstart


def join_epochs(tstart1: float, tend1: float, 
                tstart2: float, tend2: float, 
                spk: NumpyArray) -> NumpyArray:
    """
    Join spiking times from two distinct epochs as if they were continuous.

    Parameters
    ----------
    tstart1, tend1, tstart2, tend2: float
        Times boundaries of both epochs to connect (in seconds).
    spk: :obj:`mtcdb.types.NumpyArray`
        Spiking times during a *whole trial* (in seconds).
        Shape: ``(nspikes,)``.
    
    Returns
    -------
    spk_joined: :obj:`mtcdb.types.NumpyArray`
        Spiking times comprised in ``[tstart1, tend2]`` and ``[tstart2, tend2]``,
        realigned as if both epochs were continuous.
        Shape: ``(nspikes1 + nspikes2,)``.
    
    Notes
    -----
    This function is used to recompose homogeneous trials, 
    whatever the task, session, experimental parameters.
    Specifically, it allows to align the spiking times across trials.

    Examples
    --------
    - To align stimulus 'offset' across trials, the longest trials should be cropped
      by joining the periods before and after this extra-lagging window.
    - In task CLK, to keep only Clicks as stimuli, the warning TORC should be excised,
      by joining the pre-stimulus epoch and the full epoch after the warning TORC.
    - In task CLK, to keep only the warning TORCs as stimuli, the Clicks should be excised,
      by joining the epoch extenting up to the Click onset and the post-stimulus epoch.

    Implementation
    --------------
    - Extract the spiking times in both periods.
    - Shift the times in the second period by the duration of the first period.
    - Concatenate the two sets of spiking times.

    See Also
    --------
    slice_epoch: Extract spiking times within one epoch of one trial.
    """
    spk1 = slice_epoch(tstart1, tend1, spk)
    spk2 = slice_epoch(tstart2, tend2, spk) + (tend1 - tstart1)
    spk_joined = np.concatenate([spk1, spk2])
    return spk_joined


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
    

def align_trials(spk: NumpyArray) -> NumpyArray:
    """
    Align the spiking times across trials within one session.

    Parameters
    ----------
    spk: :obj:`mtcdb.types.NumpyArray`
    
    Returns
    -------
    frates: :obj:`mtcdb.types.NumpyArray`
    
    Implementation
    --------------
    - 
    """
    frates = np.array([])
    return frates


def main():
    """
    Main function for the module.
    """
    pass