#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_pseudo_trials` [module]

Classes
-------
PseudoTrialsBuilder
"""
from typing import List, Dict, Iterable

import numpy as np

from core.builders.base_builder import Builder
from core.composites.features import Features
from core.coordinates.trial_analysis_label_coord import CoordPseudoTrialsIdx, CoordFolds
from core.composites.exp_conditions import ExpCondition
from core.processors.preprocess.bootstrap import Bootstrapper
from core.processors.preprocess.map_indices import IndexMapper


class PseudoTrialsBuilder(Builder[CoordPseudoTrialsIdx]):
    """
    Build the reconstructed pseudo-trials for a pseudo-population of units.

    Product: `CoordPseudoTrialsIdx`

    The output coordinate contains the trial indices of the trials to select for each unit in one
    ensemble, for each fold and condition.
    Shape: ``(n_units, k, n_pseudo)``, with ``n_pseudo`` the total number of pseudo-trials formed
    across the experimental conditions of interest.

    The inputs required to perform this operation include, for each unit in the population, the set
    of coordinates which jointly specify the experimental condition and fold to which each trial
    belongs.

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
    TRIALS_AXIS : int
        Axis along which the trials are stored in the coordinate. Here, ``-1`` for the last axis.

    Attributes
    ----------
    counts_by_condition : Dict[ExpCondition, int]
        Number of pseudo-trials to form for one fold in each experimental condition of interest.
    order_conditions : Iterable[ExpCondition, ...]
        Order in which to concatenate the pseudo-trials for each condition in the output coordinate.

    Methods
    -------
    build (required)
    initialize_coord
    build_for_condition
    build_for_fold
    recover_indices
    gather_conditions
    gather_ensembles

    Notes
    -----
    - Trials are assigned to folds *by unit* before constructing pseudo-trials via hierarchical
      bootstrap.
    - Trials are combined *within each fold* to prevent data leakage across folds.
    - Folds assignments and bootstrapping are *stratified* by experimental condition to balance
      trial types across groups.
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
        self,
        features_by_unit: List[Features] | None = None,
        folds_by_unit: List[CoordFolds] | None = None,
        seed: int = 0,
        **kwargs,
    ) -> CoordPseudoTrialsIdx:
        """
        Implement the base class method.

        Arguments
        ---------
        features_by_unit : List[Features]
            Coordinates of the trials for each unit in the population.
            Length: ``n_units`` feature sets.
        folds_by_unit : List[CoordFolds]
            Folds labels assigned to the trials for each unit in the population.
            Length: ``n_units`` arrays. Shapes: ``(n_samples_unit,)``.
        seed : int
            Seed used by the bootstrapper.

        Returns
        -------
        pseudo_trials : PseudoTrials
            Data structure product instance.

        See Also
        --------
        `CoordPseudoTrialsIdx.from_shape`
        `CoordPseudoTrialsIdx.SENTINEL`
        """
        assert features_by_unit is not None
        assert folds_by_unit is not None
        # Initialize the coordinate for the pseudo-trials
        n_units = len(features_by_unit)
        n_pseudo_tot = sum(self.counts_by_condition.values())  # across conditions
        pseudo_trials = CoordPseudoTrialsIdx.from_shape((n_units, self.k, n_pseudo_tot))
        # Add the fold labels as features for each unit
        strata = [feat.add_coord(fold) for feat, fold in zip(features_by_unit, folds_by_unit)]
        # Bootstrap by fold and condition
        for fold in range(self.k):
            pseudo_trials_by_cond = []
            for exp_cond in self.order_conditions:
                n_pseudo = self.counts_by_condition[exp_cond]
                stratum = exp_cond.add_factor(name="fold", factor=fold)  # add fold label as feature
                # Find the indices of the trials in the fold and condition for each unit
                pseudo_trials_in_stratum = self.build_for_stratum(strata, stratum, n_pseudo, seed)
                pseudo_trials_by_cond.append(pseudo_trials_in_stratum)  # shape: (n_units, n_pseudo)
            # Gather pseudo-trials for all the conditions
            pseudo_trials_in_fold = self.gather_strata(
                pseudo_trials_by_cond, axis=self.TRIALS_AXIS, n_pseudo_tot=n_pseudo_tot
            )
            # Fill by unit
            for u in range(n_units):
                pseudo_trials[u, fold, :] = pseudo_trials_in_fold[u, :]
        self.product = pseudo_trials
        return self.get_product()

    @staticmethod
    def build_for_stratum(
        strata: List[Features], stratum: ExpCondition, n_pseudo: int, seed: int
    ) -> CoordPseudoTrialsIdx:
        """
        Build the pseudo-trials for one stratum (fold x condition).

        Arguments
        ---------
        strata : List[Features]
            Features to consider to group trials in strata for each unit in the population.
        stratum : ExpCondition
            Set of feature values defining the stratum for which to build the pseudo-trials.
        n_pseudo : int
            Number of pseudo-trials to form for this stratum.
        seed : int
            Seed used by the bootstrapper.

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            Coordinate for the pseudo-trials of the stratum. Shape: ``(n_units, n_pseudo)``.

        Implementation
        --------------
        To recover the absolute indices of the trials to select for one stratum:

        1. For each unit, identify the *absolute* indices of the trials in the stratum among all its
           (real) trials.
        2. For the population, get the *relative* indices of the trials to select within the stratum
           to form pseudo-trials for this stratum.
        3. For each unit, recover the *absolute* indices of the trials to select for the
           pseudo-trials in this stratum.

        Mapping of indices (for each unit): `idx_absolute` -> `idx_relative` -> `idx_pseudo`

        - `idx_absolute`: Indices of the trials in the stratum relative to the full set of trials.
          Shape: ``(n_samples_unit, )``.
        - `idx_relative`: Indices of the trials to select relative to the subset of trials within
          the stratum. Shape: ``(n_pseudo, )``.
        - `idx_pseudo`: Indices of the trials to select relative to the full set of trials. Shape
          ``(n_pseudo, )``.

        To obtain a matrix of indices for all the units, stack the arrays of indices for each unit.

        See Also
        --------
        `core.processors.preprocess.bootstrap.Bootstrapper`
        `core.processors.preprocess.map_indices.IndexMapper`
        `np.stack`: Stack arrays along a new axis. Here, the units axis (0).
        """
        # Find the indices and counts of the trials in the fold and condition for each unit
        idx_absolute = [coords.match_idx(stratum) for coords in strata]  # n_units arrays
        counts = [len(idx) for idx in idx_absolute]  # for Bootstrapper
        # Bootstrap across the population (indices relative to stratum)
        bootstrapper = Bootstrapper(n_pseudo)
        idx_relative = bootstrapper.process(counts=counts, seed=seed)  # shape: (n_units, n_pseudo)
        # Recover the absolute indices of the trials to select for each unit
        mapper = IndexMapper()
        idx_pseudo = [
            mapper.process(idx_abs, idx_rel) for idx_abs, idx_rel in zip(idx_absolute, idx_relative)
        ]  # n_units arrays
        # Convert to coordinate: list of n_units arrays -> array of shape (n_units, n_pseudo)
        pseudo_trials = CoordPseudoTrialsIdx(np.stack(idx_pseudo, axis=0))
        return pseudo_trials

    @staticmethod
    def gather_strata(
        pseudo_trials_by_stratum: List[CoordPseudoTrialsIdx],
        axis: int = TRIALS_AXIS,
        n_pseudo_tot: int | None = None,
    ) -> CoordPseudoTrialsIdx:
        """
        Gather the pseudo-trials for all the conditions of interest.

        Arguments
        ---------
        pseudo_trials_by_stratum : List[CoordPseudoTrialsIdx]
            Pseudo-trials for each experimental condition (within one fold).
            Length: ``n_conditions``.
            Shape of each element: ``(n_units, n_pseudo)``.
        axis : int
            Axis along which to concatenate the pseudo-trials. Here, ``-1`` for the last axis.
        n_pseudo_tot : int
            Total number of pseudo-trials to form across the experimental conditions of interest.
            Used to check the consistency of input shapes.

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            Coordinate for the pseudo-trials of all the conditions.
            Shape: ``(n_units, n_pseudo_tot)``.

        Raises
        ------
        ValueError
            If the number of pseudo-trials obtained by gathering strata does not match the total
            expected number of pseudo-trials.

        See Also
        --------
        `np.concatenate`: Concatenate arrays along a pre-existing axis, here the trials axis (-1).
        """
        # Check that number of pseudo-trials matches the total expected number
        n_pseudo = sum(pseudo_trials.shape[-1] for pseudo_trials in pseudo_trials_by_stratum)
        if n_pseudo_tot is not None and n_pseudo != n_pseudo_tot:
            raise ValueError(
                f"Mismatch in number of pseudo-trials across conditions. Actual: {n_pseudo} != Expected: {n_pseudo_tot}"
            )
        # Concatenate along the trials dimension in the right order, convert to coordinate
        pseudo_trials = CoordPseudoTrialsIdx(np.concatenate(pseudo_trials_by_stratum, axis=axis))
        return pseudo_trials

    @staticmethod
    def gather_ensembles(
        pseudo_trials_by_ensemble: List[CoordPseudoTrialsIdx],
    ) -> CoordPseudoTrialsIdx:
        """
        Gather the pseudo-trials for all the ensembles with equal number of units.

        Create a new dimension for ensembles in first position.

        Arguments
        ---------
        pseudo_trials_by_ensemble : List[CoordPseudoTrialsIdx]
            Pseudo-trials for each ensemble.

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            Coordinate for the pseudo-trials of all the ensembles.
            Shape: ``(n_ensembles, n_units, k, n_pseudo)``.

        Raises
        ------
        ValueError
            If the number of units, folds or pseudo-trials do not match across the ensembles.

        See Also
        --------
        `np.stack`: Stack arrays along a new axis, here the ensembles axis (0).
        """
        # Check that number of units, folds and pseudo-trials match across ensembles
        shapes = [pseudo_trials.shape for pseudo_trials in pseudo_trials_by_ensemble]
        if len(set(shapes)) != 1:
            raise ValueError(f"Mismatch in the shape of pseudo-trials across ensembles: {shapes}")
        # Stack along the ensemble dimension
        pseudo_trials = CoordPseudoTrialsIdx(np.stack(pseudo_trials_by_ensemble, axis=0))
        return pseudo_trials
