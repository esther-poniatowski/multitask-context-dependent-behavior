"""
`core.processors.preprocess.exclude` [module]

Classes
-------
TrialsCounter
SampleSizer
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `process` method in `TrialsCounter` and `SampleSizer`.
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------


from typing import List, Any, TypeAlias, Tuple

import numpy as np

from core.constants import N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.composites.coordinate_set import CoordinateSet
from core.processors.base_processor import Processor
from core.composites.exp_conditions import ExpCondition
from core.processors.preprocess.assign_folds import FoldAssigner
from core.processors.preprocess.bootstrap import Bootstrapper


Counts: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for the number of trials per unit."""


class TrialsCounter(Processor):
    """
    Count the number of trials available in the population for one experimental condition.

    Attributes
    ----------
    features_by_unit : List[CoordinateSet]
        Coordinates containing the factors of interest for splitting trials by condition.
        Number of elements: ``n_units``, number of units.
        Elements: ``CoordinateSet``, behaving like a list of coordinates with identical numbers of
        samples.

    Examples
    --------
    Consider a population of two units with two features (task, category):

    >>> coord_task_1 = CoordTask(['PTD', 'PTD', 'CLK'])
    >>> coord_categ_1 = CoordCategory(['T', 'T', 'R'])
    >>> coord_task_2 = CoordTask(['CLK', 'CLK', 'PTD'])
    >>> coord_categ_2 = CoordCategory(['R', 'R', 'T'])
    >>> features_1 = CoordinateSet(coord_task_1, coord_categ_1)
    >>> features_2 = CoordinateSet(coord_task_2, coord_categ_2)
    >>> features_by_unit = [features_1, features_2]

    Count the trials in a specific condition:

    >>> exp_condition = ExpCondition(Task('PTD'), Category('T'))
    >>> counter = TrialsCounter(features_by_unit)
    >>> counts = counter.process(exp_condition)
    >>> counts
    array([2, 1])

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
    """

    def __init__(self, features_by_unit: List[CoordinateSet]):
        self.features_by_unit = features_by_unit

    def process(self, exp_condition: ExpCondition) -> Counts:
        """
        Implement the abstract method of the base class `Processor`.

        Arguments
        ---------
        exp_condition : ExpCondition
            Experimental condition of interest.

        Returns
        -------
        counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
            Number of trials available for each unit in the population for the specified condition.
            Shape: ``(n_units,)``.
        """
        assert exp_condition is not None
        counts = self.count_in_condition(self.features_by_unit, exp_condition)
        return counts

    # --- Processing Methods -----------------------------------------------------------------------

    def count_in_condition(
        self, features_by_unit: List[CoordinateSet], exp_condition: ExpCondition
    ) -> Counts:
        """
        Count the number of trials available for each unit in the population.

        Arguments
        ---------
        features_by_unit : List[CoordinateSet]
            See the class attribute `features_by_unit`.
        exp_condition : ExpCondition
            See the method argument `exp_condition`.

        Returns
        -------
        counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
            See the method return value `counts`.
        """
        n_units = len(features_by_unit)
        counts = np.zeros(n_units, dtype=int)
        for i, coords in enumerate(features_by_unit):
            counts[i] = coords.count(exp_condition)
        return counts


class SampleSizer(Processor):
    """
    Determine the number of pseudo-trials to form in one condition.

    Attributes
    ----------
    n_folds : int
        Number of folds in the cross-validation.
    n_min : int
        Minimum number of trials required for one unit.
    thres_perc : float
        Percentage of the smallest count to consider for the sample size.
    """

    def __init__(
        self, n_folds: int = 1, n_min: int = N_TRIALS_MIN, thres_perc: float = BOOTSTRAP_THRES_PERC
    ) -> None:
        self.n_folds = n_folds
        self.n_min = n_min
        self.thres_perc = thres_perc

    def process(self, counts: Counts) -> int:
        """
        Implement the abstract method of the base class `Processor`.

        Arguments
        ---------
        counts : np.ndarray[Tuple[Any], np.dtype[np.int64]]
            Number of trials available for each unit in the population for the specified condition.
            Shape: ``(n_units,)``.

        Returns
        -------
        sample_size : int
            Number of pseudo-trials to form in the condition.
        """
        assert counts is not None
        sample_size = self.eval_sample_size(counts, self.n_folds, self.n_min, self.thres_perc)
        return sample_size

    # --- Processing Methods -----------------------------------------------------------------------

    @staticmethod
    def eval_sample_size(
        counts: Counts,
        n_folds: int,
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
        counts_in_fold = np.array([FoldAssigner.eval_min_count(n, n_folds) for n in counts])
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
        """
        idx_excluded = np.where(counts < sample_size)[0]
        n_excluded = len(idx_excluded)
        return n_excluded
