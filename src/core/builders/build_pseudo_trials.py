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
- Folds assignments and bootstrapping are *stratified* by condition (task, attentional state,
  stimulus, +fold for bootstrap) to balance trial types across groups. Shuffling is performed in
  those operations by the dedicated processors, to balance across folds the task variables which
  have not been considered in stratification (i.e. positional information: recording number, block
  number, slot number). This prevents models to capture misleading temporal drift in neuronal
  activity.
- Error trials can be excluded from the pseudo-trials by providing the appropriate configuration.
"""
# pylint: disable=missing-function-docstring

from types import MappingProxyType
from typing import List, Tuple, Dict, Optional, Mapping

import numpy as np

from core.builders.base_builder import DataBuilder
from core.coordinates.base_coord import Coordinate
from core.coordinates.bio_info_coord import CoordUnit
from core.coordinates.exp_factor_coord import CoordTask, CoordAttention, CoordCategory
from core.coordinates.time_coord import CoordTime
from core.data_structures.core_data import CoreData
from core.data_structures.firing_rates import FiringRatesPop, FiringRatesUnit
from core.processors.preprocess.assign_ensembles import EnsembleAssigner, Ensembles
from core.processors.preprocess.assign_folds import FoldAssigner, FoldLabels
from core.processors.preprocess.bootstrap import Bootstrapper
from core.processors.preprocess.map_indices import IndexMapper, Indices
from core.processors.preprocess.stratify import Stratifier, Strata


class PseudoTrialsBuilder(DataBuilder[FiringRatesPop]):
    """
    Build the reconstructed pseudo-trials for a pseudo-population of units.

    Product:`PseudoTrials` data structure.

    The core data in the output contains the trial indices of the trials to select from the data,
    for each unit in an ensemble and for each fold.
    Along with this core data, the product contains the coordinates for the trials dimensions, i.e.
    the experimental factors which where considered to define the conditions of interest.
    The inputs required to perform this operation include, for each unit in the population, the set
    of coordinates which jointly specify the experimental condition to which each trial belongs.

    Class Attributes
    ----------------
    PRODUCT_CLASS : type
        See the base class attribute.
    TMP_DATA : Tuple[str]
        See the base class attribute.

    Configuration Parameters
    ------------------------
    k : int
        Number of folds for cross-validation.
    n_by_cond : Dict[Tuple[str, ...], int]
        Number of pseudo-trials to generate *by condition*, common to all ensembles (determined
        beforehand).
        Keys: Experimental condition defined by a combination of features' values.
        Values: Number of pseudo-trials to generate for the condition.
    conditions : List[str]
        (Derived) Names of the conditions to consider for stratification (task, attentional state,
        stimulus).
    conditions_boundaries : Dict[str, Tuple[int, int]]
        (Derived) Start and end indices of the trials for each condition in the final data
        structure.
    features_coords : Dict[str, Coordinate]
        Nature of the features to consider for stratification.
        Keys: Names of coordinates along the trial dimension in the data structures of individual
        units. Each corresponds to a field in the `n_by_cond` dictionary.
        Values: Class of the coordinate to use for stratification.

    Processing Attributes
    ---------------------
    data_per_unit : List[FiringRatesUnit]
        Firing rates of each unit in the population.
    seed : int
        Seed for the random number generator, used in the ensemble assignment.
    n_units : int
        (Property) Number of units in the population (length of the inputs `data_per_unit`).
    n_trials : int
        (Property) Number of pseudo-trials to form (sum of the number of pseudo-trials to generate
        across conditions).

    Methods
    -------
    """

    PRODUCT_CLASS = FiringRatesPop
    TMP_DATA = ("data_per_unit", "seed", "ensembles")

    def __init__(
        self,
        n_folds: int,
        n_by_cond: Dict[str, int],
        features_coords: Mapping[str, type],
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.n_folds = n_folds
        self.n_by_cond = n_by_cond
        self.conditions = list(n_by_cond.keys())
        self.conditions_boundaries = self.allocate_conditions_indices()
        self.features_coords = features_coords
        # Declare attributes to store inputs and intermediate results
        self.data_per_unit: List[FiringRatesUnit]
        self.seed: int
        self.ensembles: Ensembles

    def build(
        self, data_per_unit: Optional[List[FiringRatesUnit]] = None, seed: int = 0, **kwargs
    ) -> FiringRatesPop:
        """
        Implement the base class method.

        Parameters
        ----------
        data_per_unit : List[FiringRatesUnit]
            See the attribute `data_per_unit`.
        seed : int
            See the attribute `seed`.

        Returns
        -------

            Data structure product instance.
        """

        return self.get_product()

    # --- Preliminary Operations -------------------------------------------------------------------

    def allocate_conditions_indices(self) -> Dict[str, Tuple[int, int]]:
        """
        Set the start and end indices of the trials of each condition in the final data structure.

        Those indices are used to ensure the consistency of the data between the core data and the
        coordinates for the trials dimension.

        Returns
        -------
        condition_boundaries : Dict[str, Tuple[int, int]]
            Start and end indices of the trials for each condition along the trials dimension in the
            final data structure.
        """
        conditions_boundaries: Dict[str, Tuple[int, int]] = {}
        start = 0
        for c in self.conditions:
            end = start + self.n_by_cond[c]
            self.conditions_boundaries[c] = (start, end)
            start = end
        return conditions_boundaries

    def stratify_by_condition(self, u_pop: int) -> Strata:
        """
        Stratify the trials of one unit by condition (task, attentional state, stimulus).

        Arguments
        ---------
        u_pop : int
            Index of the unit in the *population*. Used to retrieve the data of the unit.
            .. _u_pop:

        Returns
        -------
        strata : Strata
            Strata labels assigned to the trials of the considered unit.
            Shape: ``(n_tr_unit,)``, number of trials available for this unit.

        See Also
        --------
        `Stratifier`
        """
        strata_def = tuple(self.features_coords.keys())  # preserve order for stratum labels
        stratifier = Stratifier(strata_def=strata_def)
        fr_unit = self.data_per_unit[u_pop]  # get data from the unit
        features = [fr_unit.get_coord(name) for name in self.features_coords.keys()]
        strata = stratifier.process(features=features, return_comb=False)
        return strata

    def get_indices_in_condition(self, cond: str, strata: int) -> Indices:
        """
        Get the indices of the trials of the unit which belong to a specific condition.

        Arguments
        ---------
        cond : str
            Name of the condition to consider.
        strata : Strata
            Strata labels assigned to the trials of the unit.

        Returns
        -------
        idx_cond : Indices
            Indices of the trials of the unit which belong to the considered condition, within the
            global data set (all the trials of the unit).

        See Also
        --------
        `Stratifier.get_stratum_indices`

        `conditions` attribute: Used to get the label of the stratum, which corresponds to the index
        of the condition in the list of conditions.
        """
        stratifier = Stratifier()
        stratum_label = self.conditions.index(cond)
        idx_cond = stratifier.get_stratum_indices(stratum_label, strata)
        return idx_cond

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

    def generate_folds(self, n_samples: int, u_ens: int) -> FoldLabels:
        """
        Assign trials to folds for one unit in one ensemble, for a specific condition.

        Arguments
        ---------
        n_samples : int
            Number of trials in the subset for a condition. Used for the argument `n_samples` of the
            fold assigner.
        u_ens : int
            Index of the unit in the *ensemble*. Used for the argument `seed` of the fold assigner.

        Notes
        -----
        - Folds are contained in a list of arrays rather than in a single array to handle the
          variable number of trials per unit.
        - Use the unit's index within the ensemble as the seed to ensure that the folds are
          different for the same unit in different ensembles.
        - Use the mode "labels" to get the fold labels directly, rather than the members. Fold
          labels are then used for condition-based indexing to extract the indices of the trials in
          the fold from the subset of trials in the condition.

        Returns
        -------
        folds : FoldLabels
            Fold labels assigned to the trials of the considered unit.
            Shape: ``(n_samples,)``.
            Values: Comprised between 0 and ``n_folds - 1``.

        See Also
        --------
        `FoldAssigner`
        """
        assigner = FoldAssigner(k=self.n_folds)
        folds = assigner.process(n_samples=n_samples, seed=u_ens, mode="labels")
        return folds

    def generate_pseudo_trials(self, cond, idx_pop) -> np.ndarray:
        """
        Generate the indices of the trials to select for each unit in one ensemble, for one
        condition and one fold.

        Arguments
        ---------
        cond : str
            Name of the condition to consider.
        idx_pop : List[Indices]
            Indices of the trials of the units in the ensemble which belong to the considered fold
            and condition. Length: ``(ensemble_size,)``.

        Returns
        --------
        idx_pseudo : np.ndarray
            Indices of the trials to select for each unit in the ensemble for the considered
            condition and fold.
            Shape: ``(ensemble_size, n_pseudo_cond)``.

        See Also
        --------
        `Bootstrapper`
        `IndexMapper`
        :attr:`n_by_cond`
            Specifies the number of pseudo-trials to generate for the condition.
        :attr:`seed`
            Used to set the seed of the bootstrapper, since all the ensembles are different (no risk
            of duplicates).
        """
        # Bootstrap trial indices
        n_pseudo = self.n_by_cond[cond]
        counts = [len(idx) for idx in idx_pop]
        bootstrapper = Bootstrapper(n_pseudo=n_pseudo)
        idx_relative = bootstrapper.process(counts=counts, seed=self.seed)
        # Recover absolute indices in the global data set for each unit
        mapper = IndexMapper()
        idx_pseudo = np.ma.masked_array(np.empty_like(idx_relative), mask=True)
        # shape: (ensemble_size, n_pseudo), values: masked as long as unset
        for u_ens, idx in enumerate(idx_pop):
            idx_pseudo[u_ens] = mapper.process(idx_relative=idx_relative[u_ens], idx_absolute=idx)
        return idx_pseudo

    # --- Construct Coordinates --------------------------------------------------------------------

    def construct_trial_coords(self) -> Dict[str, Coordinate]:
        """
        Construct the trial coordinates for the pseudo-trials: task, attentional state, stimulus.

        Returns
        -------
        coords : Dict[str, Coordinate]
            Coordinates for the trial dimension in the data structure product.
        """
        # Initialize empty coordinates with as many trials as the total number of pseudo-trials
        coords = {
            name: coord_tpe(np.full((self.n_trials,), "", dtype=np.str_))
            for name, coord_tpe in self.features_coords.items()
        }
        # Fill the coordinates by condition
        for cond, (start, end) in self.conditions_boundaries.items():
            for value, name in zip(cond, self.features_coords.keys()):
                coords[name][start:end] = value
        return coords
