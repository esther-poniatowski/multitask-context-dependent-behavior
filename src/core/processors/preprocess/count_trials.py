#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.exclude` [module]

Classes
-------
`TrialsCounter`
"""

from typing import List, Any, TypeAlias, Tuple

import numpy as np

from core.constants import N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.composites.features import Features
from core.processors.base_processor import Processor
from core.composites.exp_conditions import ExpCondition
from core.processors.preprocess.assign_folds import FoldAssigner
from core.processors.preprocess.bootstrap import Bootstrapper
from core.processors.preprocess.exclude import Excluder

Counts: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for the number of trials per unit."""


class TrialsCounter(Processor):
    """
    Count the number of trials available in the population for one experimental condition.

    Attributes
    ----------
    feat_by_unit : List[Features]
        Coordinates containing the factors of interest for splitting trials by condition.
        Number of elements: ``n_units``, number of units.
        Elements: ``Features``, behaving like a list of coordinates with identical numbers of
        samples.

    Examples
    --------

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
    """

    def __init__(self, feat_by_unit: List[Features]):
        self.feat_by_unit = feat_by_unit

    def process(self, exp_cond: ExpCondition | None = None, **kwargs) -> Counts:
        """Implement the template method called in the base class `process` method.

        Arguments
        ---------
        exp_cond : ExpCondition
            Experimental condition of interest.

        Returns
        -------
        counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
            Number of trials available for each unit in the population for the specified condition.
            Shape: ``(n_units,)``.
        """
        assert exp_cond is not None
        counts = self.count_in_condition(feat_by_unit=self.feat_by_unit, exp_cond=exp_cond)
        return counts

    # --- Processing Methods -----------------------------------------------------------------------

    def count_in_condition(self, feat_by_unit: List[Features], exp_cond: ExpCondition) -> Counts:
        """
        Count the number of trials available for each unit in the population.

        Arguments
        ---------
        feat_by_unit : List[Features]
            See the class attribute `feat_by_unit`.
        exp_cond : ExpCondition
            See the method argument `exp_cond`.

        Returns
        -------
        counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
            See the method return value `counts`.
        """
        n_units = len(feat_by_unit)
        counts = np.zeros(n_units, dtype=int)
        for i, coords in enumerate(feat_by_unit):
            counts[i] = coords.count(exp_cond)
        return counts


class SampleSizer(Processor):
    """
    Determine the number of pseudo-trials to form in one condition.

    Attributes
    ----------
    counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
        Number of trials available for each unit in the population for the specified condition.
        Shape: ``(n_units,)``.
    """

    def __init__(self, counts: Counts):
        self.counts = counts

    def process(
        self,
        k: int = 1,
        n_min: int = N_TRIALS_MIN,
        thres_perc: float = BOOTSTRAP_THRES_PERC,
        **kwargs
    ):
        """
        Implement the template method called in the base class `process` method.

        Arguments
        ---------
        k : int
            Number of folds in the cross-validation.
        n_min : int
            Minimum number of trials required for one unit.
        thres_perc : float
            Percentage of the smallest count to consider for the sample size.

        Returns
        -------
        sample_size : int
            Number of pseudo-trials to form in the condition.
        """
        sample_size = self.eval_sample_size(self.counts, k, n_min, thres_perc)
        return sample_size

    # --- Processing Methods -----------------------------------------------------------------------

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
        counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
            See the configuration attribute `counts`.

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
        counts_in_fold = np.array([FoldAssigner.eval_min_count(n, k) for n in counts])
        sample_size = Bootstrapper.eval_n_pseudo(counts_in_fold, n_min, thres_perc)
        return sample_size

    @staticmethod
    def count_excluded_units(counts: Counts, sample_size: int) -> int:
        """
        Count the number of units excluded from the analysis due to insufficient trials.

        Arguments
        ---------
        counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
            See the configuration attribute `counts`.
        sample_size : int
            Number of pseudo-trials to form in the condition.

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
