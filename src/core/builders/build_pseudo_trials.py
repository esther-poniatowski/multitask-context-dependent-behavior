#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_pseudo_trials` [module]

Classes
-------
`PseudoTrialsBuilder`

"""
from typing import List, Tuple, Dict, Iterable

import numpy as np

from core.builders.base_builder import Builder
from core.composites.features import Features
from core.coordinates.trials_coord import CoordPseudoTrialsIdx
from core.composites.exp_conditions import ExpCondition
from core.processors.preprocess.assign_folds import FoldAssigner, FoldLabels
from core.processors.preprocess.bootstrap import Bootstrapper
from core.processors.preprocess.map_indices import IndexMapper, Indices


class PseudoTrialsBuilder(Builder[CoordPseudoTrialsIdx]):
    """
    Build the reconstructed pseudo-trials for a pseudo-population of units for one experimental
    condition.

    Product:`CoordPseudoTrialsIdx`

    The output coordinate contains the trial indices of the trials to select for each unit in one
    ensemble, for each fold and condition.
    Shape: ``(n_units, k, n_pseudo)``, with ``n_pseudo`` the total number of pseudo-trials formed
    across the experimental conditions of interest.

    The inputs required to perform this operation include, for each unit in the population, the set
    of coordinates which jointly specify the experimental condition to which each trial belongs.

    Warning
    -------
    The order in which the trials are stored in the output coordinate of the `PseudoTrialsBuilder`
    builder should be consistent with the other output coordinates of the `TrialCoordsBuilder`
    (which label the experimental factors of interest, see `core.builders.build_trial_coords`). To
    ensure this consistency along the trials dimension, the experimental conditions should be
    treated in the same order across both builders. This is achieved by passing them the same
    configuration parameters on which they operate:

    - `order_conditions`: Order in which the experimental conditions are concatenated.
    - `counts_by_condition`: Number of pseudo-trials to form for each experimental condition.

    Class Attributes
    ----------------
    TRiALS_AXIS : int
        Axis along which the trials are stored in the coordinate. Here, ``-1`` for the last axis.

    Attributes
    ----------
    k : int
        Number of folds for cross-validation.
    counts_by_condition : Dict[ExpCondition, int]
        Number of pseudo-trials to form for one fold in each experimental condition of interest.
    order_conditions : Iterable[ExpCondition, ...]
        Order in which to concatenate the pseudo-trials for each condition in the output coordinate.

    Methods
    -------

    Notes
    -----
    - Trials are assigned to folds *by unit* before constructing pseudo-trials via hierarchical
      bootstrap.
    - Trials are combined within each fold to prevent data leakage across folds.
    - Folds assignments and bootstrapping are *stratified* by experimental condition to balance
      trial types across groups.
    - Shuffling is performed by the dedicated processors, to balance across folds the task variables
      which have not been considered in stratification (i.e. positional information: recording
      number, block number, slot number). This prevents models from capturing misleading temporal
      drift in neuronal activity.
    """

    PRODUCT_CLASS = CoordPseudoTrialsIdx
    TRIALS_AXIS = -1

    def __init__(
        self,
        k: int,
        counts_by_condition: Dict[ExpCondition, int],
        order_conditions: Iterable[ExpCondition],
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.k = k
        self.counts_by_condition = counts_by_condition
        self.order_conditions = order_conditions

    def build(
        self, coords_by_unit: List[Features] | None = None, seed: int = 0, **kwargs
    ) -> CoordPseudoTrialsIdx:
        """
        Implement the base class method.

        Arguments
        ---------
        coords_by_unit : List[Features]
            Coordinates of the trials for each unit in the population.
        seed : int
            Seed used by the bootstrapper.

        Returns
        -------
        pseudo_trials : PseudoTrials
            Data structure product instance.
        """
        assert coords_by_unit is not None
        pseudo_trials = {
            exp_cond: self.build_for_condition(exp_cond, coords_by_unit, self.k, n_pseudo, seed)
            for exp_cond, n_pseudo in self.counts_by_condition.items()
        }
        self.product = self.gather_conditions(
            pseudo_trials, self.counts_by_condition, self.order_conditions
        )
        return self.get_product()

    @staticmethod
    def initialize_coord(n_units: int, k: int, n_pseudo: int) -> CoordPseudoTrialsIdx:
        """
        Initialize the coordinate for the pseudo-trials for one condition.

        Arguments
        ---------
        n_units : int
            Number of units in the population.
        k : int
            Number of folds for cross-validation.
        n_pseudo : int
            Number of pseudo-trials to form for one fold in this condition.

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            Empty coordinate for the pseudo-trials, filled with the sentinel value defined for the
            coordinate class.

        See Also
        --------
        `CoordPseudoTrialsIdx.from_shape`
        `CoordPseudoTrialsIdx.SENTINEL`
        """
        shape = (n_units, k, n_pseudo)
        pseudo_trials = CoordPseudoTrialsIdx.from_shape(shape)
        return pseudo_trials

    @staticmethod
    def build_for_condition(
        exp_cond: ExpCondition, coords_by_unit: List[Features], k: int, n_pseudo: int, seed: int
    ) -> CoordPseudoTrialsIdx:
        """
        Build the pseudo-trials for one experimental condition.

        Arguments
        ---------
        exp_cond : ExpCondition
            Experimental condition for which to build the pseudo-trials.
        n_pseudo : int
            Number of pseudo-trials to form for one fold in this condition.
        seed : int
            Seed used by the bootstrapper.

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            Coordinate for the pseudo-trials of the condition. Shape: ``(n_units, k, n_pseudo)``.

        Notes
        -----
        Use the unit's index within the ensemble as the seed to ensure that the folds are different
        for the same unit in different ensembles.

        See Also
        --------
        `FoldAssigner`
        """
        # Initialize the coordinate to fill, fold by fold and unit by unit
        n_units = len(coords_by_unit)
        pseudo_trials = PseudoTrialsBuilder.initialize_coord(n_units, k, n_pseudo)
        # Find the indices and counts of the trials in the condition for each unit
        idx_in_cond = [coords.match_idx(exp_cond) for coords in coords_by_unit]  # n_units arrays
        counts_in_cond = [len(idx) for idx in idx_in_cond]  # for FoldAssigner
        # Assign trials to folds for each unit
        assigner = FoldAssigner(k=k)
        folds_labels = [
            assigner.process(n_samples=n, seed=u, mode="labels")  # seed: unit index
            for u, n in enumerate(counts_in_cond)
        ]  # n_units arrays
        for fold in range(k):
            # Bootstrap by fold across the population
            idx_in_fold, idx_bootstrap = PseudoTrialsBuilder.build_for_fold(
                fold=fold, folds_labels=folds_labels, n_pseudo=n_pseudo, seed=seed
            )
            # Fill the coordinate
            for u in range(n_units):
                pseudo_trials[u, fold, :] = PseudoTrialsBuilder.recover_indices(
                    idx_in_cond=idx_in_cond[u],
                    idx_in_fold=idx_in_fold[u],
                    idx_bootstrap=idx_bootstrap[u],
                )
        return pseudo_trials

    @staticmethod
    def build_for_fold(
        fold: int,
        folds_labels: List[FoldLabels],
        n_pseudo: int,
        seed: int,
    ) -> Tuple[List[Indices], Indices]:
        """
        Build the pseudo-trials for one fold of one experimental condition.

        Arguments
        ---------
        fold_labels : List[FoldLabels]
            Fold labels assigned to the trials of each unit.
            Length: ``n_units`` arrays. Shapes: ``(n_samples_unit,)``.
        n_pseudo : int
            Number of pseudo-trials to form for this condition.

        Returns
        -------
        idx_in_fold : List[Indices]
            Indices of the trials in the fold, for each unit in the population.
        idx_bootstrap : Indices
            Indices of the trials to select for the fold, for each unit in the population.
            Shape: ``(n_units, n_pseudo)``.

        See Also
        --------
        `Bootstrapper`
        """
        # Find the indices and counts of the trials in the fold for each unit
        idx_in_fold = [np.where(labels == fold)[0] for labels in folds_labels]  # n_units arrays
        counts_in_fold = [len(idx) for idx in idx_in_fold]  # for Bootstrapper
        # Bootstrap trial indices for the population, shape: (n_units, n_pseudo)
        bootstrapper = Bootstrapper(n_pseudo=n_pseudo)
        idx_bootstrap = bootstrapper.process(counts=counts_in_fold, seed=seed)
        return idx_in_fold, idx_bootstrap

    @staticmethod
    def recover_indices(
        idx_in_cond: Indices, idx_in_fold: Indices, idx_bootstrap: Indices
    ) -> Indices:
        """
        Recover the indices of the trials to select for the pseudo-trials, for one unit.

        Arguments
        ---------
        idx_in_cond : Indices
            Indices of the trials in the experimental condition for the unit.
            Shape: ``(n_samples_unit,)``.
        idx_in_fold : Indices
            Indices of the trials in the fold for the unit. Shape: ``(n_samples_fold,)``.
        idx_bootstrap : Indices
            Indices of the trials to select for the pseudo-trials. Shape: ``(n_pseudo,)``.

        Returns
        -------
        idx_pseudo : Indices
            Indices of the trials to select for the pseudo-trials, for the unit.
            Shape: ``(n_pseudo,)``.

        Implementation
        --------------
        To recover the absolute indices of the trials to select for one fold in the experimental
        condition, several mappings of indices are performed:

        `idx_bootstrap` -> `idx_in_fold` -> `idx_in_cond`

        - `idx_in_cond`: Indices of the trials in the experimental condition.
           Shape: ``(n_samples_unit, )``.
        - `idx_in_fold`: Indices of the trials in the fold.
           Shape: ``(n_samples_fold, )``.
        - `idx_bootstrap`: Indices after bootstrapping.
           Shape ``(n_pseudo, )``.

        See Also
        --------
        `IndexMapper`
        """
        mapper = IndexMapper()
        idx_pseudo_in_fold = mapper.process(idx_absolute=idx_in_fold, idx_relative=idx_bootstrap)
        idx_pseudo = mapper.process(idx_absolute=idx_in_cond, idx_relative=idx_pseudo_in_fold)
        return idx_pseudo

    @staticmethod
    def gather_conditions(
        pseudo_trials_by_cond: Dict[ExpCondition, CoordPseudoTrialsIdx],
        counts_by_condition: Dict[ExpCondition, int],
        order_conditions: Iterable[ExpCondition],
        axis: int = TRIALS_AXIS,
    ) -> CoordPseudoTrialsIdx:
        """
        Gather the pseudo-trials for all the conditions of interest.

        Arguments
        ---------
        pseudo_trials_by_cond : Dict[ExpCondition, CoordPseudoTrialsIdx]
            Pseudo-trials for each experimental condition.
        counts_by_condition : Dict[ExpCondition, int]
            See the attribute `counts_by_condition`.
        order_conditions : Tuple[ExpCondition, ...]
            See the attribute `order_conditions`.

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            See the return value `pseudo_trials` of the `build` method.

        Raises
        ------
        ValueError
            If the number of pseudo-trials for one condition does not match the number of trials in
            the output coordinate.
        """
        # Check that number of pseudo-trials for each condition
        for exp_cond, pseudo_trials in pseudo_trials_by_cond.items():
            expected = counts_by_condition[exp_cond]
            actual = pseudo_trials.shape[axis]
            if actual != expected:
                raise ValueError(
                    f"Mismatch in the number of pseudo-trials for {exp_cond}: "
                    f"{expected} expected, {actual} in output coordinate."
                )
        # Concatenate along the trials dimension in the right order, convert to coordinate
        pseudo_trials = CoordPseudoTrialsIdx(
            np.concatenate(
                [pseudo_trials_by_cond[exp_cond] for exp_cond in order_conditions], axis=axis
            )
        )
        return pseudo_trials
