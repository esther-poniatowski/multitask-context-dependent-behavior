#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.align_spikes` [module]

Classes
-------
`SpikesAligner`

Notes
-----
In the raw data set, the distinct trials are characterized by different events and timings
(experimental variability). However, in the final data set, all the trials should be aligned to
compare the neural responses.

Each trial in the final data set should contains three epochs of fixed durations:

- Pre-stimulus period : duration ``d_pre``.
- Stimulus period : duration ``d_stim``.
- Post-stimulus period : duration ``d_post``.

Therefore, the final data set is built by recomposing homogeneous trials, whatever the task,
session, experimental parameters. To include one specific raw trial into the final data set, several
discontinuous epochs should be artificially joined.

The relevant epochs to extract depend on:

- The actual times of stimulus onset and offset in the specific trial (``t_on`` and ``t_off``).
- The type of stimulus to align and the task from which it is extracted (``stim`` and ``task``).

Stimuli might be cropped if their actual duration ``t_off - t_on`` is longer than the
duration ``d_stim`` set for the whole dataset.

Aligning stimuli from task PTD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the PTD task, one single stimulus occurs in each trial (TORC 'R' or Tone 'T').

1. Extract Pre-stimulus + Stimulus periods.
    - Start: ``t_on - d_pre`` (to align stimuli's onsets across trials)
    - End: ``t_on + d_stim``
2. Extract Post-stimulus period.
    - Start: ``t_off`` (to align stimuli's offsets across trials: true end of stimulus).
    - End: ``t_off + d_post``

Aligning stimuli from task CLK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the CLK task, two stimuli follow each other in each trial (TORC 'N' and Click train 'R'/'T').
Both should constitute independent trials in the final dataset.

Aligning a Neutral TORC
.......................
1. Extract Pre-stimulus + Stimulus periods (as for the PTD case).
2. No Post-stimulus period

Aligning a Click train
......................
1. Extract Pre-stimulus period.
- Start: ``t_on - d_pre``
- End: ``t_on`` (before the TORC)
2. Extract Stimulus + Post-stimulus periods.
- Start: ``t_on + d_warn`` (onset of the Click = offset of the TORC)
- End: ``t_start2 + d_stim + d_post`` (duration of the stimulus AND post-stimulus periods)
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member

from typing import Literal, TypeAlias, Any, Tuple, Dict

import numpy as np

from core.constants import D_PRE, D_STIM, D_POST, D_WARN
from core.processors.base_processor import Processor


Stim: TypeAlias = Literal["R", "T", "N"]
"""Type alias for stimulus type."""

Task: TypeAlias = Literal["PTD", "CLK"]
"""Type alias for task type."""

SpikingTimes: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.float64]]
"""Type alias for spiking times."""


class SpikesAligner(Processor):
    """
    Align spiking times from a single trial with the other trials in a data set.

    Provide utility functions for epoch slicing, concatenation and alignment.

    Conventions for the documentation:

    - Attributes: Configuration parameters of the processor, passed to the *constructor*.
    - Arguments: Input data to process, passed to the `process` method (base class).
    - Returns: Output data after processing, returned by the `process` method (base class).

    Class Attributes
    ----------------
    _TASKS : frozenset
        Valid types of tasks.
    _STIMS : frozenset
        Valid types of stimuli.

    Configuration Attributes
    ------------------------
    d_pre, d_stim, d_post : float
        Durations of the pre-stimulus, stimulus, and post-stimulus periods (in seconds) for all
        the trials in the final dataset.
    d_warn : float
        Duration of the TORC stimulus (in seconds), within the total trial duration in task CLK.

    Processing Arguments
    --------------------
    spikes : SpikingTimes
        Spiking times during a whole trial (in seconds). Shape: ``(n_spikes,)``.
        .. _spikes:
    task : Task
        Type of task from which the trial is extracted.
        .. _task:
    stim : Stim
        Type of stimulus presented during the trial.
        .. _stim:
    t_on, t_off : float
        Times of stimulus onset and offset (in seconds) during the *specific* trial to align.
        .. _t_on:

    Returns
    -------
    aligned_spikes : SpikingTimes
        Spiking times aligned with the other trials in the final data set.
        Shape: ``(n_spikes,)``.
        .. _aligned_spikes:

    Methods
    -------
    `eval_times`
    `slice_epoch`
    `join_epochs`

    Examples
    --------
    Align spikes with default duration parameters:

    >>> aligner = SpikesAligner()
    >>> aligned_spikes = aligner.process(spikes=spikes, task=task, stim=stim, t_on=t_on, t_off=t_off)
    >>> print(strata)
    [0 0 1]

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    _TASKS: frozenset = frozenset({"PTD", "CLK"})
    _STIMS: frozenset = frozenset({"R", "T", "N"})

    def __init__(
        self,
        d_pre: float = D_PRE,
        d_stim: float = D_STIM,
        d_post: float = D_POST,
        d_warn: float = D_WARN,
    ):
        super().__init__(d_pre=d_pre, d_stim=d_stim, d_post=d_post, d_warn=d_warn)

    def _pre_process(self, **input_data: Any) -> Dict[str, Any]:
        """
        Check that the input values for task, stimulus, and timings are consistent.

        Raises
        ------
        ValueError
            If the input values are inconsistent.
        """
        task = input_data.get("task", None)
        stim = input_data.get("stim", None)
        if task not in self._TASKS:
            raise ValueError(f"Invalid task: {task}")
        if stim not in self._STIMS:
            raise ValueError(f"Invalid stimulus: {stim}")
        return input_data

    def _process(self, **input_data: Any) -> SpikingTimes:
        """Implement the template method called in the base class `process` method."""
        spikes = input_data["spikes"]
        task = input_data["task"]
        stim = input_data["stim"]
        t_on = input_data["t_on"]
        t_off = input_data["t_off"]
        t_start1, t_end1, t_start2, t_end2 = self.eval_times(task, stim, t_on, t_off)
        spk_joined = self.join_epochs(spikes, t_start1, t_end1, t_start2, t_end2)
        return spk_joined

    def eval_times(
        self, task: Task, stim: Stim, t_on: float, t_off: float
    ) -> Tuple[float, float, float, float]:
        """
        Determine the times boundaries of the epochs to extract in one specific trial in order to
        align it with the other trials in a data set.

        Returns
        -------
        t_start1, t_end1, t_start2, t_end2 : float
            Time boundaries of the first and second epochs to extract in the specific trial.
        """
        t_start1 = t_on - self.d_pre
        if self.task == "PTD" or (task == "CLK" and stim == "N"):  # excise Click train
            t_end1 = t_on + self.d_stim
            t_start2 = t_off
            t_end2 = t_off + self.d_post
        elif self.task == "CLK" and (stim == "T" or stim == "R"):  # excise TORC
            t_end1 = t_on
            t_start2 = t_on + self.d_warn
            t_end2 = t_start2 + self.d_stim + self.d_post
        else:
            raise ValueError(f"Invalid task-stimulus combination: {task}-{stim}")
        return t_start1, t_end1, t_start2, t_end2

    def slice_epoch(self, spikes: SpikingTimes, t_start: float, t_end: float) -> SpikingTimes:
        """
        Extract spiking times within one epoch and reset them relative to the epoch start.

        Important
        ---------
        Resulting spiking times are *relative* to the beginning of the considered epoch.

        Arguments
        ---------
        spikes : SpikingTimes
            See the argument :ref:`spikes`.
        t_start, t_end: float
            Times boundaries of the epoch (in seconds).

        Returns
        -------
        spk_epoch : SpikingTimes
            Spiking times in the epoch comprised between ``t_start`` and ``t_end``, reset to be
            relative to the beginning of the epoch. Shape: ``(nspikes_epoch, 1)``.

        Implementation
        --------------
        - Select the spiking times within the epoch with a boolean mask.
        - Subtract the starting time of the epoch to reset the time.
        """
        return spikes[(spikes >= t_start) & (spikes < t_end)] - t_start

    def join_epochs(
        self, spikes: SpikingTimes, t_start1: float, t_end1: float, t_start2: float, t_end2: float
    ) -> SpikingTimes:
        """
        Join spiking times from two distinct epochs as if they were continuous.

        Parameters
        ----------
        spikes : SpikingTimes
            See the argument :ref:`spikes`.
        t_start1, t_end1, t_start2, t_end2 : float
            Times boundaries of both epochs to connect (in seconds).

        Returns
        -------
        spk_joined : SpikingTimes
            Spiking times comprised in ``[t_start1, t_end1]`` and ``[t_start2, t_end2]``, realigned
            as if both epochs were continuous. Shape: ``(nspikes1 + nspikes2,)``.

        Implementation
        --------------
        - Extract the spiking times in both epochs and reset them relative to each epoch's start.
        - Shift the times in the second period by the duration of the first period.
        - Concatenate the two sets of spiking times.
        """
        spk1 = self.slice_epoch(spikes, t_start1, t_end1)
        spk2 = self.slice_epoch(spikes, t_start2, t_end2) + (t_end1 - t_start1)
        return np.concatenate([spk1, spk2])
