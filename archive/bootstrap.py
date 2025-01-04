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

from typing import Dict, TypeAlias, Any, Tuple, List, Union, Iterable, Optional

import numpy as np

from core.constants import N_PSEUDO_MIN
from core.processors.base_processor import Processor
from core.processors.preprocess.stratify import Strata


Counts: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for the number of trials per unit."""

CountsPerStrata: TypeAlias = np.ndarray[Tuple[Any, Any], np.dtype[np.int64]]
"""Type alias for the number of trials per unit and stratum."""

TrialsIndUnit: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for trials indices for one single unit."""

StrataPop: TypeAlias = List[Strata]
"""Type alias for strata of trials per unit."""

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
    n_pseudo : Union[int, Iterable[int]]
        Total number of pseudo-trials to generate for the current run of the processor.
        If a single integer is provided, the same number of pseudo-trials is generated for all the
        stratum (or a single stratum is considered).
        If an iterable of integers is provided, each integer corresponds to the number of
        pseudo-trials to generate for each stratum. The number of labels in the strata passed to the
        `process` method must match the number of integers in `n_pseudo`.
    n_by_stratum : np.ndarray[Tuple[Any], np.dtype[np.int64]]
        [Not provided as argument, computed in the constructor from `n_pseudo`]
        Formatted representation of the number of pseudo-trials per stratum, used internally to
        handle distinct cases for the inputs.
        If `n_pseudo` is a single integer, then `n_by_stratum` is an array with a single value.
        If `n_pseudo` is an iterable of integers, then `n_by_stratum` contains the same values
        converted to a numpy array.
    n_strata : int
        [Not provided as argument, computed in the constructor from `n_by_stratum`]
        Number of strata to consider in the bootstrapping process to structure the pseudo-trials.
        If `n_pseudo` is a single integer, then `n_strata` is 1.
        If `n_pseudo` is an iterable of integers, then `n_strata` is the length of `n_by_stratum`.

    Arguments
    ---------
    counts : Counts
        Numbers of trials available for each unit in the pseudo-population.
        Shape: ``(n_units,)``.
        .. _counts:
    strata_pop : StrataPop
        Strata labels assigned to the trials for each unit in the population.
        Length: ``n_units``. Shape of each array element: ``(n_trials,)``.
        Each array corresponds to one unit and contains the stratum labels associated to each
        trial of this unit.
        .. _strata_per_unit:

    Returns
    -------
    pseudo_trials : PseudoTrials
        Indices of the trials to pick from each unit to form each pseudo-trial.
        Shape: ``(n_units, n_pseudo_tot)``.
        .. _pseudo_trials:

    Methods
    -------
    `validate_strata`
    `count_in_strata`
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

     Generate pseudo-trials structured with 2 strata, with 3 and 2 pseudo-trials respectively:

    >>> n_pseudo = [3, 2]
    >>> strata_pop = [np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([0, 1, 1])]
    >>> bootstrapper = Bootstrapper(n_pseudo=n_pseudo)
    >>> pseudo_trials = bootstrapper.process(strata_pop=strata_pop, seed=42)
    >>> print(pseudo_trials)
    [[0 1 2 0 1]  # unit 0, 2 trials in stratum 0, 1 trial in stratum 1
     [0 1 1 0 1]  # unit 1, 1 trial in stratum 0, 2 trials in stratum 1
     [0 1 1 0 1]] # unit 2, 1 trial in stratum 0, 2 trials in stratum 1

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors. See definition of class-level attributes and template
        methods.
    """

    is_random = True

    def __init__(self, n_pseudo: Union[int, Iterable[int]]) -> None:
        # Declare two new attributes
        n_by_stratum: np.ndarray[Tuple[Any], np.dtype[np.int64]]
        n_strata: int
        # Format the number of pseudo-trials per stratum
        if isinstance(n_pseudo, int):  # array with a single value
            n_by_stratum = np.array([n_pseudo])
        else:  # convert to numpy array
            n_by_stratum = np.array(n_pseudo)
        # Determine the number of strata to consider
        n_strata = len(n_by_stratum)
        super().__init__(n_pseudo=n_pseudo, n_by_stratum=n_by_stratum, n_strata=n_strata)

    # --- Input Validation and Preprocessing -------------------------------------------------------

    def _pre_process(self, **input_data: Any) -> Dict[str, Any]:
        """
        Handles the distinct cases for the inputs.

        If `n_pseudo` is a single integer:

        - Require the input `counts`.
        - Generate a default value for `strata_pop` (assuming one single strata).

        If `n_pseudo` is an iterable of integers:

        - Require the input `strata_pop`.
        - Validate its structure.

        Notes
        -----
        The bootstrapper operates only on the input data `strata_pop`. It handles the case of a
        single stratum by generating a default value for `strata_pop` based on the `counts`.
        Then, the input `counts` is not used anymore in the processing, so no default value is
        generated.
        """
        counts: Optional[Union[Counts]] = input_data.get("counts", None)
        strata_pop: Optional[StrataPop] = input_data.get("strata_pop", None)
        if self.n_strata == 1:
            if counts is not None and strata_pop is None:
                strata_pop = self.default_strata(counts)
                input_data.update({"strata_pop": strata_pop})  # only relevant input
            else:
                msg = "Invalid inputs: Pass only `counts` when `n_pseudo` is an integer."
                raise ValueError(msg)
        else:
            if strata_pop is not None and counts is None:
                self.validate_strata(strata_pop)
            else:
                msg = "Invalid inputs: Pass only `strata_pop` when `n_pseudo` is an iterable."
                raise ValueError(msg)
        return input_data

    def default_strata(self, counts: Counts) -> StrataPop:
        """
        Generate a single stratum for each unit.

        For each unit, assign the number of trials specified in `counts`, all grouped with a single
        stratum label (0).

        Arguments
        ---------
        counts : Counts
            See the argument :ref:`counts`.

        Returns
        -------
        strata_pop : StrataPop
            See the return :ref:`strata_pop`.
        """
        return [np.zeros(n, dtype=int) for n in counts]

    def validate_strata(self, strata_pop) -> None:
        """
        Validate the number of labels in the strata of each unit, where distinct labels should match
        the values in the range from 0 to `n_pseudo`

        Raises
        ------
        ValueError
            If the number of labels in the strata exceeds the number of values in `n_pseudo`.
            If any label is missing in the range from 0 to `len(n_pseudo) - 1`.
        """
        for i, strata in enumerate(strata_pop):
            labels = np.unique(strata)
            missing_labels = set(range(self.n_strata)) - set(labels)
            extra_labels = [l for l in labels if l >= self.n_strata]
            if missing_labels:
                msg = f"Missing labels: {sorted(missing_labels)} in stratum {i}."
                raise ValueError(msg)
            if extra_labels:
                msg = f"Extra labels: {extra_labels} >= {self.n_strata} in stratum {i}."
                raise ValueError(msg)

    # --- Processing -------------------------------------------------------------------------------

    def _process(self, **input_data: Any) -> PseudoTrials:
        """Implement the template method called in the base class `process` method."""
        strata_pop = input_data["strata_pop"]
        counts_per_strata = self.count_in_strata(strata_pop)
        pseudo_trials = self.bootstrap(strata_pop, counts_per_strata)
        return pseudo_trials

    def count_in_strata(self, strata_pop: StrataPop) -> CountsPerStrata:
        """
        Count the number of trials available in each stratum for each unit.

        Arguments
        ---------
        strata_pop : StrataPop
            See the argument :ref:`strata_pop`.

        Returns
        -------
        counts_per_strata : CountsPerStrata
            Number of trials available in each stratum for each unit.
            Shape: ``(n_strata, n_units)``.
            .. _counts_per_strata:

        See Also
        --------
        :func:`numpy.bincount`
            Count occurrences of each value in an array.
            Parameter `minlength`: Ensure the output array has the same length as the number of
            strata. Missing values are filled with zeros.
        """
        n_strata = len(self.n_by_stratum)
        n_units = len(strata_pop)
        counts_per_strata = np.empty((n_strata, n_units), dtype=int)
        for u, strata in enumerate(strata_pop):
            counts_per_strata[:, u] = np.bincount(strata, minlength=n_strata)
        return counts_per_strata

    def pick_trials(self, n: int, n_pseudo: int) -> TrialsIndUnit:
        """
        Pick trials's indices from a single unit for future inclusion among the pseudo-trials.

        Parameters
        ----------
        n : int
            Number of trials available for the considered unit within one stratum.
        n_pseudo : int
            Number of pseudo-trials to generate for this stratum. This is not necessarily equal to
            the attribute `n_pseudo` since it might differ across strata.
            .. _n_pseudo:

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
        Combine trials across units to form pseudo-trials within a single stratum.

        Arguments
        ---------
        counts: Counts
            Counts of available trials across units within a single stratum. Shape: ``(n_units,)``.
        n_pseudo: int
            See the argument :ref:`n_pseudo`.

        Returns
        -------
        idx : PseudoTrials
            Indices of the trials to pick within the stratum from each unit to generate the
            pseudo-trials for this stratum. For each unit, the indices are comprised between 0 and
            ``n - 1``, where ``n`` is the number of available trials in the considered stratum.
            Shape: ``(n_units, n_pseudo)``.

        Warning
        -------
        For each unit, the indices of the trials are relative to the subset of trials which
        correspond to the considered stratum. They do not necessarily match the original indices of
        those same trials among all the trials of the unit.
        """
        idx = np.array([self.pick_trials(n, n_pseudo) for n in counts])
        for trials_unit in idx:
            np.random.shuffle(trials_unit)  # shuffle within each unit to diversify pairings
        return idx

    def transpose_indices(
        self, idx_relative: TrialsIndUnit, strata: Strata, label: int
    ) -> TrialsIndUnit:
        """
        Transpose the trials' indices relative to one stratum to the 'absolute' trials' indices
        among all the trials of the considered unit (i.e. including all strata).

        Arguments
        ---------
        idx_relative : TrialsIndUnit
            Indices of trials relative to the stratum for one unit.
            Shape: ``(n_pseudo,)``, with ``n_pseudo`` the number of pseudo-trials to generate for
            this stratum.
            Values: Comprised between 0 and ``n_stratum - 1``, where ``n_stratum`` is the number of
            available trials in the considered stratum for this unit.
        strata : Strata
            Stratum labels for all the trials of the unit. Shape: ``(n_tot,)``, where ``n_tot`` is
            the total number of trials available for this unit.

        Returns
        -------
        idx_absolute : TrialsIndUnit
            Indices of the same trials among all the trials of the unit.
            Shape: ``(n_pseudo,)``.
            Values: Comprised between 0 and ``n_tot - 1``, where ``n_tot`` is the total number of
            trials available for this unit, which includes all strata.

        Implementation
        --------------
        1. Find the indices of the trials in the stratum of interest:
           ``idx_in_stratum = np.where(strata == label)[0]``
           Extract the first (unique) element of the tuple since strata is one-dimensional.
        2. Replace each relative index by the corresponding absolute index:
           ``idx_absolute = idx_in_stratum[idx_relative]``
           This generates an array with the same shape as `idx_relative`. Each value is picked from
           the array `idx_in_stratum` (absolute indices) at the index specified in `idx_relative`
           (which indeed indicates in position among the stratum).
        """
        idx_in_stratum = np.where(strata == label)[0]
        idx_absolute = idx_in_stratum[idx_relative]
        return idx_absolute

    def bootstrap(self, strata_pop: StrataPop, counts_per_strata: CountsPerStrata) -> PseudoTrials:
        """
        Bootstrap hierarchically to generate pseudo-trials structured by strata.

        Arguments
        ---------
        strata_pop : StrataPop
            See the argument :ref:`strata`.

        Returns
        -------
        pseudo_trials : PseudoTrials
            See the return :ref:`pseudo_trials`. Shape: ``(n_units, n_pseudo_tot)``.
        """
        # Initialize the pseudo-trials array
        n_units = len(strata_pop)
        n_pseudo_tot = sum(self.n_by_stratum)
        pseudo_trials = np.empty((n_units, n_pseudo_tot), dtype=int)
        # Generate pseudo-trials for each stratum (label)
        i = 0  # initial index for the pseudo-trials in the array
        for label, (n_pseudo, counts) in enumerate(zip(self.n_by_stratum, counts_per_strata)):
            # Pick and combine trials' indices *relative to the stratum*
            idx_relative = self.combine_trials(counts, n_pseudo)  # (n_units, n_pseudo)
            # Transpose to absolute indices among all the trials
            idx_absolute = np.empty((n_units, n_pseudo), dtype=int)
            for u, (idx, strata) in enumerate(zip(idx_relative, strata_pop)):
                idx_absolute[u] = self.transpose_indices(idx, strata, label)
            # Fill the array for the current stratum
            pseudo_trials[:, i : i + n_pseudo] = idx_absolute
            i += n_pseudo  # update index for next stratum
        return pseudo_trials

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
