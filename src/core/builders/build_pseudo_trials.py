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


from types import MappingProxyType
from typing import List, Tuple, Dict, Optional, Mapping, TypeVar, Type

import numpy as np

from core.builders.base_builder import Builder
from core.coordinates.base_coord import Coordinate
from core.composites.features import Features
from core.data_structures.core_data import CoreData
from core.composites.exp_conditions import ExpCondition
from core.processors.preprocess.assign_folds import FoldAssigner, FoldLabels
from core.processors.preprocess.bootstrap import Bootstrapper
from core.processors.preprocess.map_indices import IndexMapper, Indices
from core.processors.preprocess.stratify import Stratifier, Strata

PseudoTrials = TypeVar("PseudoTrials", bound=CoreData)
"""Type variable for the pseudo-trials data structure produced by the builder."""


class PseudoTrialsBuilder(Builder[CoreData]):
    """
    Build the reconstructed pseudo-trials for a pseudo-population of units for one experimental
    condition.

    Product:`PseudoTrials` data structure.

    The core data in the output contains the trial indices of the trials to select from the data,
    for each unit in one ensemble and for each fold.

    The inputs required to perform this operation include, for each unit in the population, the set
    of coordinates which jointly specify the experimental condition to which each trial belongs.

    Configuration Parameters
    ------------------------
    k : int
        Number of folds for cross-validation.
    n_pseudo : int
        Number of pseudo-trials to generate for the experimental condition in one fold.
    exp_cond : ExpCondition
        Experimental condition to consider for the pseudo-trials.

    Methods
    -------
    """

    PRODUCT_CLASS = CoreData

    def __init__(
        self,
        k: int,
        n_pseudo: int,
        exp_cond: ExpCondition,
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.k = k
        self.n_pseudo = n_pseudo
        self.exp_cond = exp_cond

    def build(
        self, coords_by_unit: List[Features] | None = None, seed: int = 0, **kwargs
    ) -> CoreData:
        """
        Implement the base class method.

        Arguments
        ---------
        coords_by_unit : List[Features]
            Coordinates of the trials for each unit in the population.
        seed : int
            Seed for the random number generator, used by the bootstrapper.

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
        n_units = len(coords_by_unit)
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
        counts_in_fold =
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
        return self.get_product()

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
