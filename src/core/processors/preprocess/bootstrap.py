#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.bootstrap` [module]

Classes
-------
:class:`Bootstrapper`

Notes
-----
Pseudo-populations of units are constructed by associating their respective trials which
were not recorded simultaneously during the experiment. To augment the resulting data set, the
current approach inspires from the "hierarchical bootstrap" method, which leverages multiple
combinations of trials. Thereby, the size of the final data set is less limited by the minimum
number of trials across neurons.

Dealing with imbalanced trial counts across units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Determination of the final number of pseudo-trials: The algorithm aims to balance the
   representation of trials across units with extreme trial counts while still maintaining a minimal
   threshold number of pseudo-trials for statistical robustness. Specifically, the goal is to avoid
   discarding too many trials from the units with numerous trials and overly duplicating those from
   the units with fewer trials.

2. Trial selection for each unit: The algorithm maximizes the diversity of the trials occurrences at
   the single unit level by ensuring that each trial is selected the maximum number of times
   possible.

3. Mitigating redundancies in trials' pairings: Trials are shuffled to obtain diverse combinations
   at the level of the pseudo-population.

Warning
-------
For data analysis:

- Assign trials to folds *by unit*.
- Assign units to batches.
- Use *stratified* assignment by condition (task, context, stimulus, error, fold, batch).

Implementation
--------------

1. Determine the number of pseudo-trials to generate based on the number of trials available for
   each unit.
2. Pick trials for each unit independently for future inclusion in the data set.
3. Combine trials across units to form actual pseudo-trials by shuffling the trials retained for
   each unit.
"""
# Disable error codes for attributes which are not detected by the type checker:
# - Configuration attributes are defined by the base class constructor.
# - Public properties for internal attributes are defined in the metaclass.
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member

from types import MappingProxyType
from typing import Optional, TypeAlias

import numpy as np
import numpy.typing as npt

from core.constants import N_PSEUDO_MIN, ALPHA_BOOTSTRAP
from core.processors.base import Processor


Counts: TypeAlias = npt.NDArray[np.int64]
"""Type alias for the number of trials per unit."""
PseudoTrials: TypeAlias = npt.NDArray[np.int64]
"""Type alias for pseudo-trials indices."""


class Bootstrapper(Processor):
    """
    Generate pseudo-trials through an algorithm inspired by the hierarchical bootstrap method.

    Attributes
    ----------
    counts: npt.NDArray[np.int_]
        Numbers of trials for each unit in the pseudo-population. Shape: ``(n_units,)``.
    n_pseudo: int, optional
        Total number of pseudo-trials to generate.
    n_global: int, optional
        Total number of pseudo-trials to generate, independently from the statistics of the trials
        count across the population.
        If provided, it serves as a global parameter which applies to all runs of the processor.
        Otherwise, the actual number of pseudo-trials is computed on each run based on the
        statistics of the trials count provided to the processor and the other configuration
        parameters (``n_pseudo_min``, ``alpha``).
    n_min: int, default=N_PSEUDO_MIN
        Minimum number of pseudo-trials required in a strata.
    alpha: float, default=ALPHA_BOOTSTRAP
        Variability factor to adjust the number of pseudo-trials to achieve sufficient diversity in
        the combinations of trials.
        Example: Setting ``alpha = 0.5`` will generate a number of pseudo-trials equal to the
        average of the minimum and maximum number of trials across units, which promotes a moderate
        level of variability.
        Example: Setting ``alpha = 0.0`` will generate a number of pseudo-trials equal to the
        minimum number of trials across units, which promotes a low level of variability.
        Example: Setting ``alpha = 1.0`` will generate a number of pseudo-trials equal to the sum of
        the minimum and maximum number of trials across units, which promotes a high level of
        variability.
    pseudo_trials : npt.NDArray[np.int_]
        Pseudo-trials obtained by pairing trials across units. It contains the indices of the real
        trials to pick from each unit to form each pseudo-trial. Shape: ``(n_units, n_pseudo)``.

    Methods
    -------
    :meth:`_pick_trials`
    :meth:`set_seed`
    :meth:`bootstrap`

    Example
    -------
    Generate pseudo-trials for three units with 10, 8, and 12 trials respectively, using a custom
    seed for reproducibility:

    >>> counts = np.array([10, 8, 12])
    >>> bootstrapper = Bootstrapper(counts, n_pseudo_min=5, alpha=0.5, seed=42)
    >>> pseudo_trials = bootstrapper.pseudo_trials # automatically computes pseudo-trials
    >>> print(pseudo_trials)
    [[ 0  1  2  3  4  5  6  7  8  9]

    Implementation
    --------------
    Private attributes used to enforce control and validation: `_counts`, `_pseudo_trials`.
    """

    config_attrs = ("n_pseudo", "n_pseudo_min", "alpha")
    input_attrs = ("counts",)
    output_attrs = ("pseudo_trials",)
    proc_data_empty = MappingProxyType(
        {
            "counts": np.array([], dtype=np.int64),
            "pseudo_trials": np.array([], dtype=np.int64),
        }
    )

    def __init__(
        self,
        n_pseudo: Optional[int] = None,
        n_pseudo_min: int = N_PSEUDO_MIN,
        alpha: float = ALPHA_BOOTSTRAP,
    ):
        super().__init__(n_pseudo=n_pseudo, n_pseudo_min=n_pseudo_min, alpha=alpha)

    def _validate(self, **input_data):
        """
        Implement the template method called in the base class :meth:`process` method.

        Raises
        ------
        ValueError
            If the `counts` attribute is invalid.
        """
        self._validate_counts(input_data["counts"])

    def _validate_counts(self, counts: Counts) -> None:
        """
        Validate the argument `counts` (number of trials per unit) based on its type and dimensions.

        Raises
        ------
        ValueError
            If the argument is not a NumPy array.
            If the argument is not 1D.
        """
        if not isinstance(counts, np.ndarray):
            raise ValueError(f"Invalid type: {type(counts)}, expected NumPy array.")
        if counts.ndim != 1:
            raise ValueError(f"Invalid dimensions: {counts.ndim}D array.")

    def _default(self, **input_data: Any) -> Dict[str, Any]:
        """
        Implement the template method called in the base class :meth:`process` method.

        Returns
        -------
        input_data : Dict[str, Any]
            Input data with default values set for missing arguments.
        """
        if self.n_pseudo is None:
            input_data["n_pseudo"] = self._compute_n_pseudo()

    def _compute_n_pseudo(self) -> int:
        """
        Determine the final number of pseudo-trials to generate from the statistics of the counts.

        Implementation
        --------------
        1. Compute a preliminary number of pseudo-trials:
           ``n_pseudo = alpha * (min_trials + max_trials)``

            - min_trials, max_trials: Minimum and maximum numbers of trials across all units
            - alpha: Variability factor to control the balance between units with few and many
              trials.

        2. Ensure the number of pseudo-trials is at least the minimum required:
           ``n_pseudo = max(n_pseudo, min_pseudo_trials)``
        """
        return max(int(self.alpha * (np.min(self.counts) + np.max(self.counts))), self.n_pseudo_min)

    @property
    def pseudo_trials(self) -> npt.NDArray[np.int_]:
        """Access the cache `_pseudo-trials` and compute it if empty."""
        if self._pseudo_trials is None:
            self._pseudo_trials = self.bootstrap()
        return self._pseudo_trials

    def _pick_trials(self, n: int) -> npt.NDArray[np.int_]:
        """
        Pick trials from a single unit for future inclusion among the pseudo-trials.

        Parameters
        ----------
        n : int
            Number of trials available for the considered unit.

        Returns
        -------
        trials : np.ndarray
            Trials selected for the considered unit.

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
        :func:`numpy.repeat`
            Repeat elements of an array, here used to selected each trial at least ``q`` times.
        """
        if n >= self.n_pseudo:
            trials = np.random.choice(n, size=self.n_pseudo, replace=False)
        else:
            q = self.n_pseudo // n  # minimal number of times each trial is selected
            r = self.n_pseudo % n  # number of trials selected additionally once
            trials = np.repeat(np.arange(n), q)
            trials = np.concatenate((trials, np.random.choice(n, size=r, replace=False)))
        return trials

    def set_seed(self) -> None:
        """
        Set the random seed for reproducibility.

        See Also
        --------
        :func:`np.random.seed`: Set the random seed for reproducibility.
        """
        np.random.seed(self.seed)

    def bootstrap(self) -> npt.NDArray:
        """
        Combine trials across units to form pseudo-trials.

        Returns
        -------
        pseudo : npt.NDArray
            See :attr:`pseudo_trials`.
        """
        self.set_seed()
        pseudo_trials = np.array([self._pick_trials(n) for n in self.counts])
        for trials_unit in pseudo_trials:
            np.random.shuffle(trials_unit)  # shuffle within each unit for diversity
        return pseudo_trials
