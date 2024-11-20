#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_pseudo_trials` [module]

Classes
-------
`PseudoTrialsBuilder`

Notes
-----
- Trials are assigned to folds *by unit*, within each ensemble, before constructing pseudo-trials
  via hierarchical bootstrap. Justification: This order of steps ensures that trials are combined
  within each fold and prevents data leakage across folds. It increases the heterogeneity of the
  data, especially in the case where two units are present in two identical ensembles. Indeed, since
  the folds of trials will be different, then the pairings of the trials are less likely match two
  identical trials from both units in both ensembles. This is possible as long as the seed varies
  for each unit when its folds are generated, which is the case if the unit does not have the same
  index in the population in both ensembles (which is likely to be the case since the units are
  shuffled when forming ensembles).
- Folds assignments and bootstrapping are *stratified* by experimental condition to balance trial
  types across groups. Shuffling is performed by the dedicated processors, to balance across folds
  the task variables which have not been considered in stratification (i.e. positional information:
  recording number, block number, slot number). This prevents models from capturing misleading
  temporal drift in neuronal activity.
"""
from typing import List, Tuple, Dict, Optional, Mapping, TypeVar, Type

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

    - `order_conds`: Order in which the experimental conditions are concatenated.
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
        Number of pseudo-trials to form for each experimental condition of interest.
    order_conds : Tuple[ExpCondition, ...]
        Order in which to concatenate the pseudo-trials for each condition in the output coordinate.

    Methods
    -------
    """

    PRODUCT_CLASS = CoordPseudoTrialsIdx
    TRIALS_AXIS = -1

    def __init__(
        self,
        k: int,
        counts_by_condition: Dict[ExpCondition, int],
        order_conditions: Tuple[ExpCondition, ...],
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.k = k
        self.counts_by_condition = counts_by_condition
        self.order_conds = order_conditions

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

        Notes
        -----
        - Use the unit's index within the ensemble as the seed to ensure that the folds are
          different for the same unit in different ensembles.
        """
        assert coords_by_unit is not None
        pseudo_trials = {
            exp_cond: self.build_for_condition(coords_by_unit, exp_cond, n_pseudo, seed)
            for exp_cond, n_pseudo in self.counts_by_condition.items()
        }
        self.product = self.gather_conditions(
            pseudo_trials, self.counts_by_condition, self.order_conds
        )
        return self.get_product()

    @staticmethod
    def initialize_coord(n_units: int, n_pseudo: int, k: int) -> CoordPseudoTrialsIdx:
        """
        Initialize the coordinate for the pseudo-trials for one condition.

        Arguments
        ---------

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            Empty coordinate for the pseudo-trials, filled with the sentinel value (here, -1).
        """
        shape = (n_units, k, n_pseudo)
        pseudo_trials = CoordPseudoTrialsIdx.from_shape(shape)
        return pseudo_trials

    @staticmethod
    def build_for_condition(
        coords_by_unit: List[Features], exp_cond: ExpCondition, n_pseudo: int, seed: int
    ) -> CoordPseudoTrialsIdx:
        """
        Build the pseudo-trials for one experimental condition.

        Arguments
        ---------
        exp_cond : ExpCondition
            Experimental condition for which to build the pseudo-trials.
        n_pseudo : int
            Number of pseudo-trials to form for this condition.
        seed : int
            Seed used by the bootstrapper.

        Returns
        -------
        pseudo_trials : CoordPseudoTrialsIdx
            Coordinate for the pseudo-trials of the condition.
        """
        # Find the indices of the trials in the condition for each unit
        # Length: n_units, Elements: Features (set of coordinates), Shape: (n_samples,) (in
        # condition, for each unit)
        idx_in_cond = [coords.match_idx(self.exp_cond) for coords in coords_by_unit]
        counts_in_cond = [len(idx) for idx in idx_in_cond]
        # Assign trials to folds for each unit
        assigner = FoldAssigner(k=self.k)
        folds_labels = [
            assigner.process(n_samples=n, seed=u, mode="labels")
            for u, n in enumerate(counts_in_cond)
        ]
        idx_in_fold = [np.where(folds_labels == fold)[0] for fold in range(self.k)]
        counts_in_fold = 0
        # Bootstrap trial indices for the population
        bootstrapper = Bootstrapper(n_pseudo=self.n_pseudo)
        idx_by_fold = [bootstrapper.process(counts=counts, seed=seed) for counts in counts_by_fold]
        # number of elements in idx_by_fold: k
        # shape of each element: (n_units, n_pseudo)
        # Recover absolute indices in the global data set for each unit
        mapper = IndexMapper()
        idx_pseudo = []
        for fold in range(self.k):
            idx_in_fold = None

        for u_ens, idx in enumerate(idx_pop):
            idx_pseudo[u_ens] = mapper.process(idx_relative=idx_relative[u_ens], idx_absolute=idx)
        return pseudo_trials

    @staticmethod
    def gather_conditions(
        pseudo_trials_by_cond: Dict[ExpCondition, CoordPseudoTrialsIdx],
        counts_by_condition: Dict[ExpCondition, int],
        order_conditions: Tuple[ExpCondition, ...],
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
            See the attribute `order_conds`.

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

    # Previous implementation ----------------------------------------------------------------------

    def get_indices_in_fold(self, fold: int, fold_labels: FoldLabels, idx: Indices) -> Indices:
        """
        Get the indices of the trials of the unit which belong to a specific fold.

        Arguments
        ---------
        fold : int
            Index of the fold to extract.
        fold_labels : FoldLabels
            Fold labels assigned to a subset of trials of the unit. Shape: ``(n_samples_subset,)``.
        idx : Indices
            Indices of a subset of trials of the unit, within the global data set (all the trials of
            the unit). Shape: ``(n_samples_subset,)``.

        Returns
        -------
        idx_fold : Indices
            Indices of the trials of the unit which belong to the considered fold, still within the
            global data set. It is a subset of the input indices.

        See Also
        --------
        `IndexMapper`
        """
        idx_relative = np.where(fold_labels == fold)[0]
        mapper = IndexMapper()
        idx_fold = mapper.process(idx_absolute=idx, idx_relative=idx_relative)
        return idx_fold
