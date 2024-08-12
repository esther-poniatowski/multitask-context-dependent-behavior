"""
:mod:`mtcdb.preprocess.firing_rates` [module]

Convert raw spike times to firing rates.

See Also
--------
test_mtcdb.test_preprocess.test_firing_rates:
    Unit tests for this module.
mtcdb.data_structures.RawSpkTimes:
    Data structure for raw spike times.

"""
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve


from core.constants import T_BIN


Stim: TypeAlias = Literal['R', 'T', 'N']
Task: TypeAlias = Literal['PTD', 'CLK']
NumpyArray: TypeAlias = npt.NDArray[np.float64]


def extract_trial(trial:int,
                  spikes: npt.NDArray[np.float64],
                  trials: npt.NDArray[np.int64]
                  ) -> npt.NDArray[np.float64]:
    """
    Extract the spiking times in one specific trial.

    Parameters
    ----------
    trial:
        Index of the trial of interest, *starting from 1*.
    spikes:
        Raw spiking times (in seconds) for one unit in a *whole session*.
        Shape: ``(nspikes,)``.
    trials:
        Trial indexes corresponding to spiking times.
        Shape: ``(nspikes,)``.

    Returns
    -------
    spk:
        Spiking times occurring in the selected trial.
        Shape: ``(nspikes_trial,)``.

    Implementation
    --------------
    To extract the spiking times of one trial,
    use a boolean mask on the trial number.

    Raises
    ------
    ValueError
        If the number of spikes and trials do not match.

    See Also
    --------
    :class:`mtcdb.data_structures.RawSpkTimes`: Data structure for raw spike times.
    """
    return spikes[trials==trial]


def slice_epoch(t_start: float, t_end: float,
                spk: NumpyArray) -> NumpyArray:
    """
    Extract spiking times within one epoch of one trial.

    Important
    ---------
    Spiking times are *relative* to the beginning of the epoch.

    Parameters
    ----------
    t_start, t_end: float
        Times boundaries of the epoch (in seconds).
    spk: :obj:`mtcdb.types.NumpyArray`
        Spiking times during a *whole trial* (in seconds).
        Shape: ``(nspikes,)``.

    Returns
    -------
    spk_epoch: :obj:`mtcdb.types.NumpyArray`
        Spiking times in the epoch comprised between ``t_start`` and ``t_end``,
        reset to be relative to the beginning of the epoch.
        Shape: ``(nspikes_epoch, 1)``.

    Implementation
    --------------
    - Select the spiking times within the epoch with a boolean mask.
    - Subtract the starting time of the epoch to reset the time.
    """
    return spk[(spk>=t_start)&(spk<t_end)] - t_start


def join_epochs(t_start1: float, t_end1: float,
                t_start2: float, t_end2: float,
                spk: NumpyArray) -> NumpyArray:
    """
    Join spiking times from two distinct epochs as if they were continuous.

    Parameters
    ----------
    t_start1, t_end1, t_start2, t_end2: float
        Times boundaries of both epochs to connect (in seconds).
    spk: :obj:`mtcdb.types.NumpyArray`
        Spiking times during a *whole trial* (in seconds).
        Shape: ``(nspikes,)``.

    Returns
    -------
    spk_joined: :obj:`mtcdb.types.NumpyArray`
        Spiking times comprised in ``[t_start1, t_end2]`` and ``[t_start2, t_end2]``,
        realigned as if both epochs were continuous.
        Shape: ``(nspikes1 + nspikes2,)``.

    Notes
    -----
    This function is used to recompose homogeneous trials,
    whatever the task, session, experimental parameters.
    Specifically, it allows to align the spiking times across trials.

    Implementation
    --------------
    - Extract the spiking times in both periods.
    - Shift the times in the second period by the duration of the first period.
    - Concatenate the two sets of spiking times.

    See Also
    --------
    slice_epoch: Extract spiking times within one epoch of one trial.
    """
    spk1 = slice_epoch(t_start1, t_end1, spk)
    spk2 = slice_epoch(t_start2, t_end2, spk) + (t_end1 - t_start1)
    spk_joined = np.concatenate([spk1, spk2])
    return spk_joined


def align_timings(task: Task, stim: Stim,
                  d_pre: float,
                  d_stim: float,
                  d_post: float,
                  d_warn: float,
                  t_on: float,
                  t_off: float
                  ) -> tuple[float, float, float, float]:
    """
    Determine the times boundaries of the epochs to extract in one trial.

    The goal is to align *all* trials across tasks and stimuli.

    Parameters
    ----------
    task: Task {'PTD', 'CLK'}
        Type of task.
    stim: Stim {'R', 'T', 'N'}
        Type of stimulus.
    d_pre, d_stim, d_post: float
        Durations of the pre-stimulus, stimulus,
        and post-stimulus periods (in seconds),
        common to *all* trials in the final dataset.
    d_warn: float
        Duration of the TORC stimulus (in seconds),
        within the total trial duration in task CLK.
    t_on, t_off: float
        Times of stimulus onset and offset (in seconds)
        during the *specific* trial.

    Returns
    -------
    t_start1, t_end1, t_start2, t_end2: tuple[float, float, float, float]
        Time boundaries of the first and second epochs to extract
        in the specific trial.

    Notes
    -----
    Each trial in the final data set contains three epochs,
    whose durations are common across all trials :

    - Pre-stimulus period : duration ``d_pre``.
    - Stimulus period : duration ``d_stim``.
    - Post-stimulus period : duration ``d_post``.

    To do so, in each specific trial from the raw data, several discontinuous
    epochs should be joined artificially.
    The relevant epochs to extract depend on :

    - The actual times of stimulus onset and offset in the specific trial,
      which may vary across sessions and trials (experimental variability).
    - The type of task and stimulus to align.

    In task PTD, one single stimulus occurs in one trial
    (TORC 'R' or Tone 'T').
    In task CLK, two stimuli follow each other in one trial
    (TORC 'N' and Click train 'R'/'T').
    Both should constitute independent trials in the final dataset.
    To do so, the other stimulus should be excised from the epoch.

    Implementation
    --------------
    **Task PTD or Task CLK with TORC**

    The first retained epoch encompasses pre-stimulus *and* stimulus periods.
    To align all the stimuli's onsets across trials, this epoch should start
    a duration ``d_pre`` before the stimulus onset,
    i.e. at time ``t_on - d_pre``.
    To keep a common stimulus duration across trials, this epoch should end
    a duration ``d_stim`` after the stimulus onset,
    i.e. at time ``t_on + d_stim``.
    The second retained epoch encompasses only the post-stimulus period.
    To align all the stimuli's offsets across trials, this epoch should start
    at the true end of stimulus ``t_off``, and should end
    at a duration ``d_post`` after the stimulus offset,
    i.e. at time ``t_off + d_post``.

    **Task CLK with Click**

    The first retained epoch encompasses only the pre-stimulus period.
    It should start as for the PTD case.
    It should end before the TORC, i.e. at time ``t_on``.
    The second retained epoch encompasses both the stimulus and post-stimulus periods.
    It should start at the beginning of the Click, i.e. at the offset of the TORC,
    i.e. at time ``t_on + d_warn``.
    It should last the duration of the stimulus AND post-stimulus periods,
    i.e. end at ``t_start2 + d_stim + d_post``.

    .. note::
        Stimuli might be cropped if their actual duration ``t_off - t_on``
        is longer than the duration ``d_stim`` set for the whole dataset.

    Raises
    ------
    ValueError
        If the task or stimulus is unknown.

    See Also
    --------
    join_epochs: Join spiking times from two distinct epochs as if they were continuous.
    """
    t_start1 = t_on - d_pre
    if task == 'PTD' or (task == 'CLK' and stim == 'N'): # excise Click train
        t_end1 = t_on + d_stim
        t_start2 = t_off
        t_end2 = t_off + d_post
    elif task == 'CLK' and (stim == 'T' or stim == 'R'): # excise TORC
        t_end1 = t_on
        t_start2 = t_on + d_warn
        t_end2 = t_start2 + d_stim + d_post
    else:
        raise ValueError("Unknown task or stimulus")
    return t_start1, t_end1, t_start2, t_end2


def spikes_to_rates(spk: np.ndarray,
                    t_bin: float,
                    t_max: float,
                    ) -> NumpyArray:
    """
    Convert a spike train into a firing rate time course.

    Parameters
    ----------
    spk: :obj:`mtcdb.types.ArrayLike`
        Spiking times.
    t_bin: float
        Time bin (in seconds).
    t_max: float
        Duration of the recording period (in seconds).

    Returns
    -------
    frates: :obj:`mtcdb.types.NumpyArray`
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
    frates = np.histogram(spk, bins=np.arange(0, t_max+t_bin, t_bin))[0]/t_bin
    frates = frates[:,np.newaxis] # add one dimension for trials
    return frates


def smooth(frates: NumpyArray,
           window: float,
           t_bin: float,
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
    t_bin: float
        Time bin (in seconds).

    Returns
    -------
    smoothed: :obj:`mtcdb.types.NumpyArray`
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
    kernel = np.ones((int(window/t_bin), 1)) # add one dimension for shape compatibility
    smoothed = fftconvolve(frates, kernel, mode=mode, axes=0)/len(kernel)
    return smoothed


def main():
    """
    Process all the raw data of one neuron to compute its final firing rates.
    """
    pass

###############################################################################
###############################################################################
