"""
:mod:`mtcdb.preprocess.firing_rates` [module]
=============================================

Convert raw spike times to firing rates.
"""

import numpy as np
from scipy.signal import fftconvolve
from typing import Any

from mtcdb.constants import TBIN
from mtcdb.types import ArrayLike, NumpyArray


def spikes_to_rates(spk: ArrayLike,
                    tbin: float,
                    tmax: float,
                    ) -> NumpyArray:
    """
    Convert a spike train into a firing rate time course.
    
    Parameters
    ----------
    spk: ArrayLike
        Spiking times.
    tbin: float
        Time bin (in seconds).
    tmax: float
        Duration of the recording period (in seconds).
    
    Returns
    -------
    frates: NumpyArray
        Firing rate time course (in spikes/s).
        Shape: ``(ntpts, 1)`` with ``ntpts = tmax/tbin`` (number of bins).
    
    See Also
    --------
    numpy.histogram: 
        Used to count the number of spikes in each bin.
        Input: ``bin`` (bin edges, *including the rightmost edge*).
        Outputs: ``hist`` (number of spikes in each bin), ``edges`` (useless).
    
    Notes
    -----

    Algorithm

    - Divide the recording period  ``[0, tmax]`` into bins of size ``tbin``.
    - Count the number of spikes in each bin.
    - Divide the spikes count in each bin by the bin size ``tbin``.

    [Op1] Bin edges are obtained with :func:`numpy.arange`, 
    with the last bin edge at ``tmax + tbin`` to include the last bin.
    
    [Op2] The output shape is has two dimensions representing
    time (length ``n_bins``),
    trials (length ``1``, single trial).
    It ensures compatibility and consistence in the full process.
    """
    frates = np.histogram(spk, bins=np.arange(0, tmax+tbin, tbin))[0]/tbin # [Op1]
    frates = frates[:,np.newaxis] # [Op2]
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
    frates: NumpyArray
        Firing rate time course (in spikes/s).
        Shape: ``(ntpts, ntrials)``,
    window: float
        Smoothing window size (in seconds).
    tbin: float
        Time bin (in seconds).
    
    Returns
    -------
    smoothed: NumpyArray
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

    Convolution Modes: ``'full'`` / ``'valid'`` / ``'same'``

    - ``'same'``: Keep the output shape as the input sequence.
    - ``'valid'``: Keep only the values which are not influenced by zero-padding.
    """
    kernel = np.ones((int(window/tbin), 1)) # add one dimension for shape compatibility
    smoothed = fftconvolve(frates, kernel, mode=mode, axes=0)/len(kernel)
    return smoothed
    
