"""
:mod:`core.pipelines.preprocess.firing_rates` [module]

Convert raw spike times to firing rates.
"""

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve


from core.constants import T_BIN


Stim: TypeAlias = Literal["R", "T", "N"]
Task: TypeAlias = Literal["PTD", "CLK"]
NumpyArray: TypeAlias = npt.NDArray[np.float64]



def align_timings(
    task: Task,
    stim: Stim,
    d_pre: float,
    d_stim: float,
    d_post: float,
    d_warn: float,
    t_on: float,
    t_off: float,
) -> tuple[float, float, float, float]:
    
    t_start1 = t_on - d_pre
    if task == "PTD" or (task == "CLK" and stim == "N"):  # excise Click train
        t_end1 = t_on + d_stim
        t_start2 = t_off
        t_end2 = t_off + d_post
    elif task == "CLK" and (stim == "T" or stim == "R"):  # excise TORC
        t_end1 = t_on
        t_start2 = t_on + d_warn
        t_end2 = t_start2 + d_stim + d_post
    else:
        raise ValueError("Unknown task or stimulus")
    return t_start1, t_end1, t_start2, t_end2


def spikes_to_rates(
    spk: np.ndarray,
    t_bin: float,
    t_max: float,
) -> NumpyArray:
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


def smooth(
    frates: NumpyArray,
    window: float,
    t_bin: float,
    mode: str = "valid",
) -> NumpyArray:
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


def main():
    """
    Process all the raw data of one neuron to compute its final firing rates.
    """
    pass


###############################################################################
###############################################################################
