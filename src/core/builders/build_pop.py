#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_pop` [module]

Classes
-------
`PopulationBuilder`

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
- Folds assignments and bootstrapping are *stratified* by condition (task, context, stimulus, +fold
  for bootstrap) to balance trial types across groups. Shuffling is performed in those operations by
  the dedicated processors, to balance across folds the task variables which have not been
  considered in stratification (i.e. positional information: recording number, block number, slot
  number). This prevents models to capture misleading temporal drift in neuronal activity.
- The product data structure is initialized at the beginning of the `build()` method for incremental
  construction. Each step can immediately update the product as soon as a component becomes
  available, without requiring temporary storage.
- The structure of trials' epochs to store in the time axis can be recovered from the data
  structures of the single units (``t_max``, ``t_on``, ``t_off``, ``t_shock``, ``t_bin``). The
  parameter ``smooth_window`` is not kept since it is used to convert to rates at the previous
  step (i.e. to build the `FiringRatesUnit` data structures).
- Error trials can be excluded from the pseudo-trials. This is done in place on the input data.
"""
# pylint: disable=missing-function-docstring

from types import MappingProxyType
from typing import List, Tuple, Dict, Optional, Mapping

import numpy as np

from core.builders.base_builder import DataBuilder
from core.coordinates.base_coord import Coordinate
from core.coordinates.bio import CoordUnit
from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim
from core.coordinates.time import CoordTime
from core.data_structures.core_data import CoreData
from core.data_structures.firing_rates import FiringRatesPop, FiringRatesUnit
from core.processors.preprocess.assign_ensembles import EnsembleAssigner, Ensembles
from core.processors.preprocess.assign_folds import FoldAssigner, FoldLabels
from core.processors.preprocess.bootstrap import Bootstrapper
from core.processors.preprocess.map_indices import IndexMapper, Indices
from core.processors.preprocess.stratify import Stratifier, Strata


class PopulationBuilder(DataBuilder[List[FiringRatesUnit], FiringRatesPop]):
    """
    Build a `FiringRatesPop` data structure.

    - Inputs: List of data structures for a set of units in a population.
    - Output: Reconstructed pseudo-trials in this pseudo-population.

    Detailed inputs and outputs are documented in the `build` method.

    Class Attributes
    ----------------
    product_class : type
        See the base class attribute.
    TMP_DATA : Tuple[str]
        See the base class attribute.
    DEFAULT_FEATURES : Mapping[str, Coordinate]
        Default features to consider for stratification. See the attribute `features_coords`.

    Configuration Parameters
    ------------------------
    ensemble_size : int
        Number of units required to form each ensemble (imposed by the area with the lowest number
        of units).
    n_ensembles_max : int
        Maximum number of ensembles to form from the population (to limit the number of ensembles
        and computational cost for the areas with the largest populations).
    k : int
        Number of folds for cross-validation.
    n_by_cond : Dict[Tuple[str, ...], int]
        Number of pseudo-trials to generate *by condition*, common to all ensembles (determined
        beforehand).
        Keys: Experimental condition defined by a combination of features' values.
        Values: Number of pseudo-trials to generate for the condition.
    conditions : List[str]
        (Derived) Names of the conditions to consider for stratification (task, context, stimulus).
    conditions_boundaries : Dict[str, Tuple[int, int]]
        (Derived) Start and end indices of the trials for each condition in the final data
        structure.
    features_coords: Dict[str, Coordinate]
        Nature of the features to consider for stratification.
        Keys: Names of coordinates along the trial dimension in the data structures of individual
        units. Each corresponds to a field in the `n_by_cond` dictionary.
        Values: Class of the coordinate to use for stratification.
    has_errors : bool, default=False
      Whether to include error trials in the pseudo-trials.

    Processing Attributes
    ---------------------
    data_per_unit : List[FiringRatesUnit]
        Firing rates of each unit in the population.
    seed : int
        Seed for the random number generator, used in the ensemble assignment.
    ensembles : Ensembles
        Indices of the units in each ensemble. Shape: ``(n_ensembles, ensemble_size)``.
    n_ensembles : int
        (Property) Number of ensembles to form.
    n_units : int
        (Property) Number of units in the population (length of the inputs `data_per_unit`).
    n_trials : int
        (Property) Number of pseudo-trials to form (sum of the number of pseudo-trials to generate
        across conditions).
    n_t : int
        Number of time points (from the common time axis across units).

    Methods
    -------
    `allocate_conditions_indices`
    `exclude_errors`
    `generate_ensembles`
    `construct_core_data`
    `initialize_core_data`
    `stratify_by_condition`
    `get_indices_in_condition`
    `get_indices_in_fold`
    `generate_folds`
    `generate_pseudo_trials`
    `fill_data`
    `construct_units_coord`
    `construct_trial_coords`
    `construct_time_coord`
    `build` (implementation of the base class method)
    """

    product_class = FiringRatesPop
    TMP_DATA = ("data_per_unit", "seed", "ensembles")
    DEFAULT_FEATURES = MappingProxyType({"task": CoordTask, "ctx": CoordCtx, "stim": CoordStim})

    def __init__(
        self,
        ensemble_size: int,
        n_ensembles_max: int,
        n_folds: int,
        n_by_cond: Dict[str, int],
        features_coords: Mapping[str, Coordinate] = DEFAULT_FEATURES,
        has_errors: bool = False,
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.ensemble_size = ensemble_size
        self.n_ensembles_max = n_ensembles_max
        self.n_folds = n_folds
        self.n_by_cond = n_by_cond
        self.conditions = list(n_by_cond.keys())
        self.conditions_boundaries = self.allocate_conditions_indices()
        self.features_coords = features_coords
        self.has_errors = has_errors
        # Declare attributes to store inputs and intermediate results
        self.data_per_unit: List[FiringRatesUnit]
        self.seed: int
        self.ensembles: Ensembles

    def build(
        self,
        area: Optional[str] = None,
        training: Optional[bool] = None,
        data_per_unit: Optional[List[FiringRatesUnit]] = None,
        seed: int = 0,
        **kwargs
    ) -> FiringRatesPop:
        """
        Implement the base class method.

        Parameters
        ----------
        area : str
            Brain area from which the units were recorded.
        training : bool
            Whether the units comes from trained or naive animals.
        data_per_unit : List[FiringRatesUnit]
            See the attribute `data_per_unit`.
        seed : int
            See the attribute `seed`.

        Returns
        -------
        FiringRatesPop
            Data structure product instance.
        """
        assert data_per_unit is not None and area is not None and training is not None
        # Store inputs
        self.data_per_unit = data_per_unit
        self.seed = seed
        # Preliminary set up
        if not self.has_errors:
            self.data_per_unit = self.exclude_errors()
        self.ensembles = self.generate_ensembles()
        # Initialize the data structure with its metadata (base method)
        self.initialize_data_structure(area=area, training=training)
        # Add core data values to the data structure
        data = self.construct_core_data()
        self.add_data(data)
        # Add coordinates in the data structure
        units = self.construct_units_coord()
        self.add_coords(units=units)
        trial_coords = self.construct_trial_coords()
        self.add_coords(**trial_coords)
        time = self.construct_time_coord()
        self.add_coords(time=time)
        return self.get_product()

    # --- Shape and Dimensions ---------------------------------------------------------------------

    @property
    def n_units(self) -> int:
        return len(self.data_per_unit)

    @property
    def n_ensembles(self) -> int:
        return EnsembleAssigner.eval_n_ensembles(
            self.n_units, self.ensemble_size, self.n_ensembles_max
        )

    @property
    def n_trials(self) -> int:
        return sum(self.n_by_cond.values())

    @property
    def n_t(self) -> int:
        return self.data_per_unit[0].get_size("time")

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
        conditions_boundaries = {}
        start = 0
        for c in self.conditions:
            end = start + self.n_by_cond[c]
            self.conditions_boundaries[c] = (start, end)
            start = end
        return conditions_boundaries

    def exclude_errors(self) -> None:
        """
        Exclude error trials from the firing rates of each unit.

        Returns
        -------
        List[FiringRatesUnit]
            New data structure instances for the population, where error trials have been excluded.
        """
        return [fr_unit.exclude_errors() for fr_unit in self.data_per_unit]

    def generate_ensembles(self) -> None:
        """
        Generate the ensembles of units to form the pseudo-population.

        Returns
        -------
        ensembles : Ensembles
            See the attribute generated by this method.

        See Also
        --------
        :meth:`EnsembleAssigner.assign`
        """
        assigner = EnsembleAssigner(self.ensemble_size, self.n_ensembles_max)
        ensembles = assigner.process(n_units=self.n_units, seed=self.seed)
        return ensembles

    # --- Construct Core Data ----------------------------------------------------------------------

    def construct_core_data(self) -> CoreData:
        """
        Construct the core data array containing the firing rates of the pseudo-population.

        Returns
        -------
        data : CoreData
            Data array containing the firing rates of the pseudo-population in pseudo-trials.
            Shape: ``(n_ensembles, ensemble_size, n_folds, n_trials, n_t)``.
            .. _data:

        Implementation
        --------------
        1. Initialize the core data array with empty values.
        2. Process each ensemble independently:
            1. For each unit in the ensemble, stratify the trials by condition.
            2. Process each condition independently, to preserve the indices of each condition in
               the final data structure along the trials dimension (i.e. to match the indices of the
               other components constructed by the method `construct_trial_coords`).
                1. For each unit, extract the indices of the trials in the considered condition.
                2. For each unit, assign folds to the trials within the considered condition.
                3. Process each fold independently within the condition:
                    1. For each unit, generate the indices of the trials within the considered fold
                       (within the condition).
                    2. For the *whole ensemble*, generate the indices of the trials to select.
                    3. Fill the data array with the selected trials at the appropriate indices for
                       the considered fold and condition.

        See Also
        --------
        :meth:`initialize_core_data`: Parent method to initialize an empty core data array.
        """
        shape = (self.n_ensembles, self.ensemble_size, self.n_folds, self.n_trials, self.n_t)
        data = self.initialize_core_data(shape)  # parent method
        for ens, units in enumerate(self.ensembles):
            cond_pop = [self.stratify_by_condition(u_pop) for u_pop in units]
            for cond in self.conditions:
                idx_cond = [self.get_indices_in_condition(cond, strata) for strata in cond_pop]
                folds_pop = [
                    self.generate_folds(len(idx), u_ens) for u_ens, idx in enumerate(idx_cond)
                ]
                for fold in range(self.n_folds):
                    idx_cond_x_fold = [
                        self.get_indices_in_fold(fold, fold_labels, idx)
                        for fold_labels, idx in zip(folds_pop, idx_cond)
                    ]
                    idx_pseudo = self.generate_pseudo_trials(cond, idx_cond_x_fold)
                    start, end = self.conditions_boundaries[cond]
                    idx_final = np.arange(start, end)
                    self.fill_data(data, ens, fold, idx_init=idx_pseudo, idx_final=idx_final)
        return data

    def stratify_by_condition(self, u_pop: int) -> Strata:
        """
        Stratify the trials of one unit by condition (task, context, stimulus).

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
        :meth:`Stratifier.process`
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
        :meth:`Stratifier.get_stratum_indices`

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
        :meth:`IndexMapper.process`
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
        :meth:`FoldAssigner.process`
        """
        assigner = FoldAssigner(k=self.n_folds)
        folds = assigner.process(n_samples=n_samples, seed=u_ens, mode="labels")
        return folds

    def generate_pseudo_trials(self, cond, idx_pop) -> None:
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
        :meth:`Bootstrapper.process`
        :meth:`IndexMapper.process`
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

    def fill_data(
        self, data: CoreData, ens: int, fold: int, idx_init: np.ndarray, idx_final: np.ndarray
    ):
        """
        Fill a part of the data array with the firing rates of the pseudo-population.

        Arguments
        ---------
        data : CoreData
            See the return :ref:`data`.
        ens : int
            Index of the ensemble.
        fold : int
            Index of the fold.
        idx_init : np.ndarray
            Indices of the trials to select in the global data set for each unit.
            Shape: ``(ensemble_size, n_trials)``, with ``n_trials`` the number of trials to select.
        idx_final : np.ndarray
            Indices in the final data array where the trials should be stored.
            Shape: ``(n_trials,)``, with ``n_trials`` equal to the shape of the last dimension of
            the argument `idx_init`.

        See Also
        --------
        :func:`np.put_along_axis`
            Fill a part of an array with values along a specified axis. Here, it is used to avoid
            mixing fancy indexing (with ``idx_final``) with basic indexing (with ``ens``, ``u_ens``,
            ``fold``, and slice ``:`` for the time axis). The argument ``axis=0`` specifies that the
            insertion happens along the trial dimension since it operates on the portion ``data[ens,
            u_ens, fold]``. The argument ``idx_final[:, None]`` is used to broadcast the indices to
            a 2D array to match the shape of the values to insert.
        """
        units = self.ensembles[ens]  # get the units in the ensemble
        for u_ens, u_pop in enumerate(units):  # u_ens: in ensemble, u_pop: in population
            rates = self.data_per_unit[u_pop].data[idx_init[u_ens]]
            np.put_along_axis(data[ens, u_ens, fold], idx_final[:, None], rates, axis=0)

    # --- Construct Coordinates --------------------------------------------------------------------

    def construct_units_coord(self) -> CoordUnit:
        """
        Construct the units coordinate for the pseudo-population (unit labels in each ensemble).

        Returns
        -------
        units : CoordUnit
            See the attribute `units` in the data structure product.

        Warning
        -------
        Two indices are used to identify the units in the population:

        - ``unit``: index of the unit in the population (from 0 to ``n_units``).
        - ``u``: index of the unit in the ensemble (from 0 to ``ensemble_size``).
        """
        units = CoordUnit(np.full((self.n_ensembles, self.ensemble_size), "", dtype=np.str_))
        for ens, units_in_ensemble in enumerate(self.ensembles):
            for u, unit in units_in_ensemble:
                units[ens, u] = self.data_per_unit[unit].name
        return units

    def construct_trial_coords(self) -> Dict[str, Coordinate]:
        """
        Construct the trial coordinates for the pseudo-trials: task, context, stimulus.

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

    def construct_time_coord(self) -> CoordTime:
        """
        Construct the time coordinate, if all the units share the same time axis.

        Returns
        -------
        time : CoordTime
            See the attribute `time` in the data structure product.
        """
        time_axis = self.data_per_unit[0].get_coord("time")  # time axis of the first unit
        for fr_unit in self.data_per_unit[1:]:  # check consistency with other units
            assert np.array_equal(fr_unit.get_coord("time"), time_axis)
        time = CoordTime(time_axis)
        return time
