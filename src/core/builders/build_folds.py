#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_folds` [module]

Classes
-------
FoldsBuilder
"""
from typing import Iterable

from core.builders.base_builder import Builder
from core.composites.coordinate_set import CoordinateSet
from core.coordinates.trial_analysis_label_coord import CoordFolds
from core.composites.exp_conditions import ExpCondition
from core.processors.preprocess.assign_folds import FoldAssigner


class FoldsBuilder(Builder[CoordFolds]):
    """
    Build the folds for cross-validation, stratified by experimental condition.

    Product: `CoordFolds`

    The output coordinate contains the folds assignments for one unit in one ensemble, gathering all
    the experimental conditions of interest.
    Shape: ``(n_trials,)``, with ``n_trials`` the total number of trials available for the unit.

    The inputs required to perform this operation include, for each unit in the population, the set
    of coordinates which jointly specify the experimental condition to which each trial belongs.

    Warning
    -------
    The order in which the trials are stored in the output coordinate of the `FoldsBuilder` builder
    should be consistent with the other output coordinates of the `TrialCoordsBuilder` (which label
    the experimental factors of interest, see `core.builders.build_trial_coords`). To ensure this
    consistency along the trials dimension, the experimental conditions should be treated in the
    same order across both builders. This is achieved by passing them the same configuration
    parameters on which they operate: `order_conditions`.

    Attributes
    ----------
    n_folds : int
        Number of folds for cross-validation.
    order_conditions : Iterable[ExpCondition, ...]
        Order in which to concatenate the pseudo-trials for each condition in the output coordinate.

    Methods
    -------
    build (required)

    Notes
    -----
    - Folds assignments and bootstrapping are *stratified* by experimental condition to balance
      trial types across groups.
    - Shuffling is performed to balance across groups the task variables which have not been
      considered in stratification (i.e. positional information: recording number, block number,
      slot number). This prevents models from capturing misleading temporal drift in neuronal
      activity.
    """

    PRODUCT_CLASS = CoordFolds

    def __init__(
        self,
        n_folds: int,
        order_conditions: Iterable[ExpCondition],
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.n_folds = n_folds
        self.order_conditions = order_conditions

    def build(
        self,
        features: CoordinateSet | None = None,
        seed: int = 0,
        **kwargs,
    ) -> CoordFolds:
        """
        Implement the base class method.

        Arguments
        ---------
        features : CoordinateSet
            Coordinates for the all the trials of the considered unit.
        seed : int
            Seed used by the FoldAssigner.

        Returns
        -------
        pseudo_trials : PseudoTrials
            Data structure product instance.
        """
        assert features is not None
        # Initialize empty coordinate
        n_trials = features.n_samples
        folds_labels = CoordFolds.from_shape((n_trials,))
        # Build folds by condition
        assigner = FoldAssigner(n_folds=self.n_folds)
        for exp_cond in self.order_conditions:  # ensure correct order
            idx = features.match_idx(exp_cond)  # indices of the trials in the condition
            n_samples = len(idx)  # counts for FoldAssigner
            folds_in_cond = assigner.process(n_samples, seed=seed, mode="labels")
            folds_labels[idx] = folds_in_cond  # fill at the indices of the condition
        return self.get_product()
