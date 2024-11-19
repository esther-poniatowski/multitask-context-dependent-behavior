#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.exclude` [module]

Classes
-------
`TrialsCounter`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# pylint: disable=useless-parent-delegation

from typing import List, Any, TypeAlias, Tuple

import numpy as np

from core.constants import N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.coordinates.coord_manager import CoordManager
from core.processors.base_processor import Processor
from core.entities.exp_conditions import ExpCondition
from core.processors.preprocess.assign_folds import FoldAssigner
from core.processors.preprocess.bootstrap import Bootstrapper
from core.processors.preprocess.exclude import Excluder

Counts: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for the number of trials per unit."""


class TrialsCounter(Processor):
    """
    Count the number of trials available in the population for one experimental condition.

    Configuration Attributes
    ------------------------
    coords_by_unit : List[CoordManager]
        Coordinates containing the factors of interest for splitting trials by condition.
        Number of elements: ``n_units``, number of units.
        Elements: ``CoordManager``, behaving like a list of coordinates with identical numbers of
        samples.

    Processing Arguments
    --------------------
    exp_cond : ExpCondition
        Experimental condition of interest.

    Returns
    -------
    counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
        Number of trials available for each unit in the population for the specified condition.
        Shape: ``(n_units,)``.

    Examples
    --------

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    IS_RANDOM = False

    # --- Processing Methods -----------------------------------------------------------------------

    def __init__(self, coords_by_unit: List[CoordManager]):
        super().__init__()
        self.coords_by_unit = coords_by_unit

    def _process(self, exp_cond: ExpCondition | None = None, **input_data) -> Counts:
        """Implement the template method called in the base class `process` method."""
        assert exp_cond is not None
        counts = self.count_in_condition(coords_by_unit=self.coords_by_unit, exp_cond=exp_cond)
        return counts

    def count_in_condition(
        self, coords_by_unit: List[CoordManager], exp_cond: ExpCondition
    ) -> Counts:
        """Count the number of trials available for each unit in the population."""
        n_units = len(coords_by_unit)
        counts = np.zeros(n_units, dtype=int)
        for i, coords in enumerate(coords_by_unit):
            counts[i] = coords.count(exp_cond)
        return counts


class SampleSizer(Processor):
    """
    Determine the number of pseudo-trials to form in one condition.

    Configuration Attributes
    ------------------------
    counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
        Number of trials available for each unit in the population for the specified condition.
        Shape: ``(n_units,)``.

    Processing Arguments
    --------------------
    k : int
        Number of folds in the cross-validation.
    n_min : int
        Minimum number of trials required for one unit.
    thres_perc : float
        Percentage of the smallest count to consider for the sample size.
    """

    def __init__(self, counts: Counts):
        super().__init__()
        self.counts = counts

    def _process(
        self,
        k: int = 1,
        n_min: int = N_TRIALS_MIN,
        thres_perc: float = BOOTSTRAP_THRES_PERC,
        **kwargs
    ):
        """Implement the template method called in the base class `process` method."""
        sample_size = self.eval_sample_size(self.counts, k, n_min, thres_perc)
        return sample_size

    @staticmethod
    def eval_sample_size(
        counts: Counts,
        k: int,
        n_min: int = N_TRIALS_MIN,
        thres_perc: float = BOOTSTRAP_THRES_PERC,
    ) -> int:
        """
        Compute the number of pseudo-trials to form in the condition.

        Arguments
        ---------

        Implementation
        --------------
        1. Find the minimum number of trials available in a fold for each unit (after folds
           assignment).
        2. Select the smallest count as the sample size.

        See Also
        --------
        `FoldAssigner.eval_min_count`
        `Bootstrapper.eval_n_pseudo`
        """
        counts_in_fold = np.array([FoldAssigner.eval_min_count(k, n) for n in counts])
        sample_size = Bootstrapper.eval_n_pseudo(counts_in_fold, n_min, thres_perc)
        return sample_size

    @staticmethod
    def count_excluded_units(counts: Counts, sample_size: int) -> int:
        """
        Count the number of units excluded from the analysis due to insufficient trials.

        Arguments
        ---------

        Returns
        -------
        n_excluded : int
            Number of units excluded from the analysis.

        See Also
        --------
        `Excluder.exclude_from_counts`
        """
        retained = Excluder.exclude_from_counts(counts, sample_size)
        n_excluded = len(counts) - len(retained)
        return n_excluded
