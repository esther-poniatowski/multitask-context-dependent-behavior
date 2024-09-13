#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.align_spikes` [module]

Classes
-------
:class:`SpikesAligner`

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

from typing import Literal, TypeAlias, Optional

import numpy as np
import numpy.typing as npt

from core.constants import D_PRE, D_STIM, D_POST, D_WARN

Stim: TypeAlias = Literal["R", "T", "N"]
"""Type alias for stimulus type."""
Task: TypeAlias = Literal["PTD", "CLK"]
"""Type alias for task type."""
SpikingTimes: TypeAlias = npt.NDArray[np.float64]
"""Type alias for spiking times."""


class SpikesAligner:
    """
    Align spiking times from a single trial with the other trials in a data set.

    Provide utility functions for epoch slicing, concatenation and alignment.

    Attributes
    ----------
    d_pre, d_stim, d_post: float
        Durations of the pre-stimulus, stimulus, and post-stimulus periods (in seconds) for all
        the trials in the final dataset.
    d_warn: float
        Duration of the TORC stimulus (in seconds), within the total trial duration in task CLK.
    spikes: npt.NDArray[np.float64]
        Spiking times during a whole trial (in seconds). Shape: ``(nspikes,)``.
    """

    valid_tasks = {"PTD", "CLK"}
    valid_stims = {"R", "T", "N"}

    def __init__(
        self,
        d_pre: float = D_PRE,
        d_stim: float = D_STIM,
        d_post: float = D_POST,
        d_warn: float = D_WARN,
        spikes: Optional[SpikingTimes] = None,
        task: Optional[Task] = None,
        stim: Optional[Stim] = None,
        t_on: Optional[float] = None,
        t_off: Optional[float] = None,
    ):
        # Set immutable attributes
        self._d_pre = d_pre
        self._d_stim = d_stim
        self._d_post = d_post
        self._d_warn = d_warn
        # Set mutable attributes
        self.spikes = spikes
        self.task = task
        self.stim = stim
        self.t_on = t_on
        self.t_off = t_off

    def _validate_exp_params(self, task: Task, stim: Stim) -> None:
        """
        Check that the input values for task, stimulus, and timings are consistent.

        Raises
        ------
        ValueError
            If the input values are inconsistent.
        """
        if task not in self.valid_tasks:
            raise ValueError(f"Unknown task: {task}")
        if stim not in self.valid_stims:
            raise ValueError(f"Unknown stimulus: {stim}")

    def slice_epoch(self, t_start: float, t_end: float) -> SpikingTimes:
        """
        Extract spiking times within one epoch of one trial.

        Important
        ---------
        Resulting spiking times are *relative* to the beginning of the considered epoch.

        Parameters
        ----------
        t_start, t_end: float
            Times boundaries of the epoch (in seconds).

        Returns
        -------
        spk_epoch: npt.NDArray[np.float64]
            Spiking times in the epoch comprised between ``t_start`` and ``t_end``, reset to be
            relative to the beginning of the epoch. Shape: ``(nspikes_epoch, 1)``.

        Implementation
        --------------
        - Select the spiking times within the epoch with a boolean mask.
        - Subtract the starting time of the epoch to reset the time.
        """
        return self.spikes[(self.spikes >= t_start) & (self.spikes < t_end)] - t_start

    def join_epochs(
        self, t_start1: float, t_end1: float, t_start2: float, t_end2: float
    ) -> SpikingTimes:
        """
        Join spiking times from two distinct epochs as if they were continuous.

        Parameters
        ----------
        t_start1, t_end1, t_start2, t_end2: float
            Times boundaries of both epochs to connect (in seconds).

        Returns
        -------
        spk_joined: npt.NDArray[np.float64]
            Spiking times comprised in ``[t_start1, t_end2]`` and ``[t_start2, t_end2]``, realigned
            as if both epochs were continuous. Shape: ``(nspikes1 + nspikes2,)``.

        Implementation
        --------------
        - Extract the spiking times in both periods.
        - Shift the times in the second period by the duration of the first period.
        - Concatenate the two sets of spiking times.
        """
        spk1 = self.slice_epoch(t_start1, t_end1)
        spk2 = self.slice_epoch(t_start2, t_end2) + (t_end1 - t_start1)
        return np.concatenate([spk1, spk2])

    def align(
        self,
    ) -> tuple[float, float, float, float]:
        """
        Determine the times boundaries of the epochs to extract in one specific trial in order to
        align it with the other trials in a data set.

        Parameters
        ----------
        task: {'PTD', 'CLK'}
            Type of task.
        stim: {'R', 'T', 'N'}
            Type of stimulus.

        t_on, t_off: float
            Times of stimulus onset and offset (in seconds) during the *specific* trial.

        Returns
        -------
        t_start1, t_end1, t_start2, t_end2: tuple[float, float, float, float]
            Time boundaries of the first and second epochs to extract in the specific trial.

        Raises
        ------
        ValueError
            If the task or stimulus is unknown.
        """
        t_start1 = self.t_on - self._d_pre
        if self.task == "PTD" or (self.task == "CLK" and self.stim == "N"):  # excise Click train
            t_end1 = self.t_on + self._d_stim
            t_start2 = self.t_off
            t_end2 = self.t_off + self._d_post
        elif self.task == "CLK" and (self.stim == "T" or self.stim == "R"):  # excise TORC
            t_end1 = self.t_on
            t_start2 = self.t_on + self._d_warn
            t_end2 = t_start2 + self._d_stim + self._d_post
        return t_start1, t_end1, t_start2, t_end2
