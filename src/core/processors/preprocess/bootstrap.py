#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.bootstrap` [module]

Classes
-------
Bootstrapper

Notes
-----
Pseudo-populations of units are constructed by associating their respective trials which
were not recorded simultaneously during the experiment. To augment the resulting data set, the
current approach inspires from the "hierarchical bootstrap" method, which leverages multiple
combinations of trials. Thereby, the size of the final data set is less limited by the minimum
number of trials across neurons.

Addressing imbalanced trial counts across units:

- Setting the number of pseudo-trials to generate: To achieve statistical robustness, the number of
  pseudo-trials is determined based on the distribution of trials counts across the population. The
  algorithm aims to balance two objectives: minimizing trial discards for units with high trial
  counts and limiting trial duplication for units with low trial counts.
- Maximizing diversity at single unit level: Each trial is included as many times as possible within
  the constraints of the balancing procedure.
- Maximizing diversity at the pseudo-population level: Trials are shuffled during the pairing
  process create diverse combinations, thereby mitigating redundancy.

Implementation
--------------

1. Determine the number of pseudo-trials to generate based on the number of trials available for
   each unit.
2. Pick trials for each unit independently for future inclusion in the data set.
3. Combine trials across units to form actual pseudo-trials by shuffling the trials retained for
   each unit.
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `process` method in `Bootstrapper`.
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------

from typing import TypeAlias, Any, Tuple

import numpy as np

from core.constants import N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.processors.base_processor import Processor, set_random_state


Counts: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for the number of trials per unit."""

TrialsIndUnit: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for trials indices for one single unit."""

PseudoTrials: TypeAlias = np.ndarray[Tuple[Any, Any], np.dtype[np.int64]]
"""Type alias for pseudo-trials indices."""


class Bootstrapper(Processor):
    """
    Generate pseudo-trials through an algorithm inspired by the hierarchical bootstrap method.

    Attributes
    ----------
    n_pseudo : int
        Total number of pseudo-trials to generate for the current run of the processor.

    Methods
    -------
    `pick_trials`
    `combine_trials`
    `bootstrap`
    `eval_n_pseudo`

    Example
    -------
    Generate pseudo-trials for three units with 10, 8, and 12 trials respectively, using a custom
    seed for reproducibility:

    >>> n_pseudo = 5
    >>> counts = np.array([4, 5, 6])
    >>> bootstrapper = Bootstrapper(n_pseudo=n_pseudo)
    >>> pseudo_trials = bootstrapper.process(counts=counts, seed=42)
    >>> print(pseudo_trials)
    [[0 1 2 3 0]  # unit 0, 4 initial trials
     [0 1 2 3 4]  # unit 1, 5 initial trials
     [0 1 2 3 4]] # unit 2, 6 initial trials

    Warning
    -------
    For each unit, the indices of the trials might be *relative* to a subset of trials in the global
    data set (e.g. one stratum). They do not necessarily match the original indices of those same
    trials among all the trials of the unit.

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
    """

    def __init__(self, n_pseudo: int) -> None:
        self.n_pseudo = n_pseudo

    @set_random_state
    def process(self, counts: Counts, seed: int = 0) -> PseudoTrials:
        """
        Implement the abstract method called in the base class `process` method.

        Arguments
        ----------
        counts : Counts
            Numbers of trials available for each unit in the pseudo-population.
            Shape: ``(n_units,)``.

        Returns
        -------
        pseudo_trials : PseudoTrials
            Indices of the trials to pick from each unit to form each pseudo-trial.
            Shape: ``(n_units, n_pseudo)``.
        """
        assert counts is not None
        pseudo_trials = self.combine_trials(counts, self.n_pseudo)
        return pseudo_trials

    # --- Processing Methods -----------------------------------------------------------------------

    @staticmethod
    @set_random_state
    def pick_trials(n: int, n_pseudo: int, seed: int = 0) -> TrialsIndUnit:
        """
        Pick trials's indices from a single unit for future inclusion among the pseudo-trials.

        Parameters
        ----------
        n : int
            Number of trials available for the considered unit.
        n_pseudo : int
            Number of pseudo-trials to generate.
        seed : int, default=0
            Seed for reproducibility.

        Returns
        -------
        idx : TrialsIndUnit
            Trials indices selected for the considered unit. Shape: ``(n_pseudo,)``. Indices are
            comprised between 0 and ``n - 1``.

        Notes
        -----
        Each trial might occur multiple times or not at all depending on the number of available
        trials for the unit (``n``) compared to the required number pseudo-trials to generate
        (``n_pseudo``).

        - If `n >= n_pseudo``: Trials are randomly selected without replacement. It includes the
          case ``n == n_pseudo``, where each trial is selected only once.
        - If ``n < n_pseudo``: Each trial is selected at least ``n_pseudo // n`` times. To reach the
          required number of trials, ``n_pseudo % n`` remaining trials are randomly selected without
          replacement.

        See Also
        --------
        :func:`numpy.random.choice`
            Randomly select elements from an array.
            Parameter `replace=False`: Prevent duplicates.
            Output: Array of selected elements (or single element if `size=1`).
        :fun:`numpy.atleast_1d`
            Convert input to an array with at least one dimension.
            Here is is used to ensure that `trials` is a 1D array in case the parameter
            `n_pseudo`=1, for shape consistency with the pseudo-trials array.
        :func:`numpy.repeat`
            Repeat elements of an array, here used to selected each trial at least ``q`` times.
        """
        if n >= n_pseudo:
            idx = np.atleast_1d(np.random.choice(n, size=n_pseudo, replace=False))
        else:
            q = n_pseudo // n  # minimal number of times each trial is selected
            r = n_pseudo % n  # number of trials selected additionally once
            idx = np.repeat(np.arange(n), q)
            idx = np.concatenate((idx, np.random.choice(n, size=r, replace=False)))
        return idx

    @staticmethod
    @set_random_state
    def combine_trials(counts: Counts, n_pseudo: int, seed: int = 0) -> PseudoTrials:
        """
        Combine trials across units to form pseudo-trials.

        Arguments
        ---------
        counts: Counts
            Counts of available trials across units. Shape: ``(n_units,)``.
        n_pseudo: int
            See the argument :ref:`n_pseudo`.

        Returns
        -------
        idx : PseudoTrials
            Indices of the trials to pick from each unit to generate the pseudo-trials. For each
            unit, the indices are comprised between 0 and ``n - 1``, where ``n`` is the number of
            available trials.
            Shape: ``(n_units, n_pseudo)``.
        """
        idx = np.array([Bootstrapper.pick_trials(n, n_pseudo) for n in counts])
        for trials_unit in idx:
            np.random.shuffle(trials_unit)  # shuffle within each unit to diversify pairings
        return idx

    # --- Utility Methods --------------------------------------------------------------------------

    @staticmethod
    def eval_n_pseudo(
        counts: Counts, n_min: int = N_TRIALS_MIN, thres_perc: float = BOOTSTRAP_THRES_PERC
    ) -> int:
        """
        Determine a number of pseudo-trials to generate from the statistics of the counts.

        The number of trials is adjusted to an arbitrary percentile of the distribution of counts,
        once the units with fewer trials than the minimum required are excluded. It aims to balance
        the representation of trials across units with extreme trial counts while still maintaining
        a minimal number of pseudo-trials for statistical robustness.

        Arguments
        ---------
        counts : Counts
            See the argument `counts` in the `process` method.
        n_min : int, default=N_PSEUDO_MIN
            Minimum number of pseudo-trials required for a unit to be included in the
            pseudo-population.
        thres_perc : float, default=0.3
            Threshold percentile of the distribution of counts at which the number of pseudo-trials
            is set.

        Returns
        -------
        n_pseudo : int
            Number of pseudo-trials to generate based on the statistics of the counts.

        Examples
        --------
        Effect of the parameter `thres_perc` on the number of pseudo-trials to generate:

        - ``0.0`` -> As many pseudo-trials as the unit with the fewest trials. For all the units
          with more trials, several trials are discarded.
        - ``1.0`` -> As many pseudo-trials as the unit with the most trials. For all the other
          units, several trials are selected multiple times.
        - ``0.5`` -> As many pseudo-trials as the median number of trials across units. For about
          half of the units, several trials are discarded, and for the other half, several are
          selected multiple times.

        Raises
        ------
        ValueError
            If no unit has enough trials for the minimum required.

        Implementation
        --------------
        1. Exclude the counts which are below the minimum required.
        2. Determine the desired percentile of the distribution of counts.

        Notes
        -----
        This method is not involved during the actual bootstrap processing. It can be used in
        preliminary exploration of the data set to determine the number of pseudo-trials to generate
        in subsequent processing.
        """
        # Consider only the units with more than the minimum number of trials
        counts = counts[counts >= n_min]  # shape: (n_units_retained,)
        if counts.size == 0:
            raise ValueError(f"No unit has enough trials for `n_min`={n_min}.")
        # Determine the percentile of the distribution of counts
        n_pseudo = int(np.percentile(counts, 100 * thres_perc))
        return n_pseudo
