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
    
    See Also
    --------
    numpy.histogram: 
        Count the number of spikes in each bin.
        Parameter ``bin``: Bin edges, *including the rightmost edge*.
        Two outputs: ``hist`` (number of spikes in each bin), ``edges``.
    
    Notes
    -----

    Algorithm

    - Divide the recording period  [0, ``tmax``] into bins of size ``tbin``.
    - Count the number of spikes in each bin.
    - Divide the number of spikes in each bin by the bin size ``tbin``.

    Bin edges are obtained with :func:`numpy.arange`, 
    with the last bin edge at ``tmax + tbin`` for the last bin to be included.
    """
    frates = np.histogram(spk, bins=np.arange(0, tmax+tbin, tbin))[0]/tbin
    return frates


def smooth(frates: Any,
           window: float,
           tbin: float,
           ) -> Any:
    """
    Smooth the firing rates in time.

    See Also
    --------
    scipy.signal.fftconvolve
    """
    pass
    
