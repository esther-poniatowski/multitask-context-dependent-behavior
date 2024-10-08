#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.bootstrap` [module]

Classes
-------
`Bootstrapper`

Notes
-----
Pseudo-populations of units are constructed by associating their respective trials which
were not recorded simultaneously during the experiment. To augment the resulting data set, the
current approach inspires from the "hierarchical bootstrap" method, which leverages multiple
combinations of trials. Thereby, the size of the final data set is less limited by the minimum
number of trials across neurons.

Dealing with imbalanced trial counts across units:

- Determination of the final number of pseudo-trials: The algorithm aims to balance the
  representation of trials across units with extreme trial counts while still maintaining a minimal
  threshold number of pseudo-trials for statistical robustness. Specifically, the goal is to avoid
  discarding too many trials from the units with numerous trials and overly duplicating those from
  the units with fewer trials.
- Trial selection for each unit: The algorithm maximizes the diversity of the trials occurrences at
  the single unit level by ensuring that each trial is selected the maximum number of times
  possible.
- Mitigating redundancies in trials' pairings: Trials are shuffled to obtain diverse combinations at
  the level of the pseudo-population.

Implementation
--------------

1. Determine the number of pseudo-trials to generate based on the number of trials available for
   each unit.
2. Pick trials for each unit independently for future inclusion in the data set.
3. Combine trials across units to form actual pseudo-trials by shuffling the trials retained for
   each unit.
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member

from typing import TypeAlias, Any, Tuple

import numpy as np

from core.constants import N_PSEUDO_MIN
from core.processors.base_processor import Processor


Counts: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for the number of trials per unit."""

TrialsIndUnit: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for trials indices for one single unit."""

PseudoTrials: TypeAlias = np.ndarray[Tuple[Any, Any], np.dtype[np.int64]]
"""Type alias for pseudo-trials indices."""


class Bootstrapper(Processor):
    """
    Generate pseudo-trials through an algorithm inspired by the hierarchical bootstrap method.

    Conventions for the documentation:

    - Attributes: Configuration parameters of the processor, including those passed to the
      *constructor* AND additional configurations automatically computed from the latter.
    - Arguments: Input data to process, passed to the `process` method (base class).
    - Returns: Output data after processing, returned by the `process` method (base class).

    Attributes
    ----------
    n_pseudo : int
        Total number of pseudo-trials to generate for the current run of the processor.

    Arguments
    ---------
    counts : Counts
        Numbers of trials available for each unit in the pseudo-population.
        Shape: ``(n_units,)``.
        .. _counts:

    Returns
    -------
    pseudo_trials : PseudoTrials
        Indices of the trials to pick from each unit to form each pseudo-trial.
        Shape: ``(n_units, n_pseudo)``.
        .. _pseudo_trials:

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
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random = True

    def __init__(self, n_pseudo: int) -> None:
        super().__init__(n_pseudo=n_pseudo)

    def _process(self, **input_data: Any) -> PseudoTrials:
        """Implement the template method called in the base class `process` method."""
        counts = input_data["counts"]
        pseudo_trials = self.combine_trials(counts, self.n_pseudo)
        return pseudo_trials

    # --- Processing Methods -----------------------------------------------------------------------

    def pick_trials(self, n: int, n_pseudo: int) -> TrialsIndUnit:
        """
        Pick trials's indices from a single unit for future inclusion among the pseudo-trials.

        Parameters
        ----------
        n : int
            Number of trials available for the considered unit.
        n_pseudo : int
            Number of pseudo-trials to generate.

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

    def combine_trials(self, counts: Counts, n_pseudo: int) -> PseudoTrials:
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
        idx = np.array([self.pick_trials(n, n_pseudo) for n in counts])
        for trials_unit in idx:
            np.random.shuffle(trials_unit)  # shuffle within each unit to diversify pairings
        return idx

    # --- Utility Methods --------------------------------------------------------------------------

    @staticmethod
    def eval_n_pseudo(counts: Counts, n_min: int = N_PSEUDO_MIN, alpha: float = 0.5) -> int:
        """
        Determine a number of pseudo-trials to generate from the statistics of the counts.

        Arguments
        ---------
        counts : Counts
            See the argument :ref:`counts`.
        n_min : int, default=N_PSEUDO_MIN
            Minimum number of pseudo-trials required in a strata.
        alpha : float, default=0.5
            Variability factor to adjust the number of pseudo-trials to achieve sufficient diversity
            in the combinations of trials.

        Returns
        -------
        n_pseudo : int
            Number of pseudo-trials to generate based on the statistics of the counts.

        Examples
        --------
        - Setting ``alpha = 0.5`` will generate a number of pseudo-trials equal to the average of
          the minimum and maximum number of trials across units, which promotes a moderate
            level of variability.
        - Setting ``alpha = 0.0`` will generate a number of pseudo-trials equal to the minimum
          number of trials across units, which promotes a low level of variability.
        - Setting ``alpha = 1.0`` will generate a number of pseudo-trials equal to the sum of the
          minimum and maximum number of trials across units, which promotes a high level of
          variability. However, this is not recommended as it may lead to overfitting.

        Implementation
        --------------
        1. Compute a preliminary number of pseudo-trials:
           ``n_pseudo = alpha * (min_count + max_count)``

            - min_count, max_count: Minimum and maximum numbers of trials across all units
            - alpha: Variability factor to control the balance between units with few and many
              trials.

        2. Ensure the number of pseudo-trials is at least the minimum required:
           ``n_pseudo = max(n_pseudo, n_min)``

        Notes
        -----
        This method is not involved during processing. It can be used in preliminary exploration of
        the data set to determine the number of pseudo-trials to generate in subsequent processing.
        """
        return max(int(alpha * (np.min(counts) + np.max(counts))), n_min)
