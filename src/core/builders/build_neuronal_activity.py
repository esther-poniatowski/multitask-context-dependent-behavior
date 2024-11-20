#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_neuronal_activity` [module]

# TODO: This is the place where to pass options such as computing firing rates, normalization...

Classes
-------
`NeuronalActivityBuilder`

Notes
-----
- The structure of trials' epochs to store in the time axis can be recovered from the data
  structures of the single units (``t_max``, ``t_on``, ``t_off``, ``t_shock``, ``t_bin``). The
  parameter ``smooth_window`` is not kept since it is used to convert to rates at the previous
  step (i.e. to build the `FiringRatesUnit` data structures).
"""
# pylint: disable=missing-function-docstring

from types import MappingProxyType
from typing import List, Tuple, Dict, Optional, Mapping

import numpy as np

from core.builders.base_builder import DataStructureBuilder
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


class FiringRatesPopBuilder(DataStructureBuilder[FiringRatesPop]):
    """
    Build the firing rate activity of a pseudo-population of units in selected pseudo-trials.

    Product: `FiringRatesPop` data structure.

    Class Attributes
    ----------------
    PRODUCT_CLASS : type
        See the base class attribute.
    TMP_DATA : Tuple[str]
        See the base class attribute.

    Configuration Parameters
    ------------------------
    ensemble_size : int
        Number of units required to form each ensemble (imposed by the area with the lowest number
        of units).
    k : int
        Number of folds for cross-validation.

    Processing Attributes
    ---------------------
    data_per_unit : List[FiringRatesUnit]
        Firing rates of each unit in the population.
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
    """

    PRODUCT_CLASS = FiringRatesPop
    TMP_DATA = ("data_per_unit", "seed", "ensembles")
    DEFAULT_FEATURES = MappingProxyType(
        {"task": CoordTask, "attn": CoordAttention, "categ": CoordCategory}
    )

    def __init__(
        self,
        ensemble_size: int,
        n_ensembles_max: int,
        n_folds: int,
        counts_by_condition: Dict[str, int],
        features_coords: Mapping[str, type] = DEFAULT_FEATURES,
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.ensemble_size = ensemble_size
        self.n_ensembles_max = n_ensembles_max
        self.n_folds = n_folds
        self.counts_by_condition = counts_by_condition
        self.conditions = list(counts_by_condition.keys())
        self.conditions_boundaries = self.allocate_conditions_indices()
        self.features_coords = features_coords
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
        return sum(self.counts_by_condition.values())

    @property
    def n_t(self) -> int:
        return self.data_per_unit[0].get_size("time")

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
        `initialize_core_data`: Parent method to initialize an empty core data array.
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
