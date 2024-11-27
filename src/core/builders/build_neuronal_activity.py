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

from typing import List, Dict

import numpy as np

from core.builders.base_builder import Builder
from core.coordinates.time_coord import CoordTime
from core.data_structures.core_data import CoreData
from core.data_structures.spike_times_raw import SpikeTimesRaw
from core.coordinates.trial_analysis_label_coord import CoordPseudoTrialsIdx


class FiringRatesBuilder(Builder[CoreData]):
    """
    Build the firing rate activity of units in selected pseudo-trials.

    Product: `CoreData`

    Attributes
    ----------
    ensemble_size : int
        Number of units required to form each ensemble (imposed by the area with the lowest number
        of units).
    n_folds : int
        Number of folds for cross-validation.


    Methods
    -------
    """

    PRODUCT_CLASS = CoreData

    def __init__(
        self,
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters

    def build(
        self,
        spikes: SpikeTimesRaw | None = None,
        pseudo_trials_idx: CoordPseudoTrialsIdx | None = None,
        **kwargs
    ) -> CoreData:
        """
        Implement the base class method.

        Parameters
        ----------
        spikes : SpikeTimesRaw
            Spiking times of one unit in the pseudo-population.
        pseudo_trials_idx : CoordPseudoTrialsIdx
            Coordinate of the pseudo-trials indices.
            Shape: ``(n_folds, n_pseudo)``.

        Returns
        -------
        product : CoreData
            Firing rates of the pseudo-population in pseudo-trials, i.e. actual values to analyze.
        """
        assert spikes is not None
        assert pseudo_trials_idx is not None
        # Initialize the data structure
        n_folds, n_pseudo = pseudo_trials_idx.shape
        shape = (n_folds, n_pseudo)
        data = CoreData.from_shape(shape, dims=("folds", "trials"))
        # Fill the data structure

        # Add core data values to the data structure
        product = self.construct_core_data()
        return self.get_product()

    # --- Construct Core Data ----------------------------------------------------------------------

    @staticmethod
    def initialize_data(
        n_ensembles: int,
        ensemble_size: int,
        n_folds: int,
        n_trials: int,
        n_t: int,
    ) -> CoreData:
        """
        Initialize the core data to be build: set its shape and dimensions.

        Arguments
        ---------
        n_ensembles : int
            Number of ensembles.
        ensemble_size : int
            Number of units in each ensemble.
        n_folds : int
            Number of folds for cross-validation.
        n_trials : int
            Number of trials in the pseudo-population.
        n_t : int
            Number of time points in a trial.

        Returns
        -------
        data : CoreData
            Empty data array to store the firing rates of the pseudo-population.
            Shape: ``(n_ensembles, ensemble_size, n_folds, n_trials, n_t)``.
            Values: Empty values (``np.nan``).

        See Also
        --------
        `CoreData.from_shape`
        """
        shape = (n_ensembles, ensemble_size, n_folds, n_trials, n_t)
        dims = ("ensemble", "unit", "fold", "trial", "time")
        data = CoreData.from_shape(shape, dims)
        return data

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


class TimeCoordBuilder(CoordTime):
    """
    Build the time coordinate for the firing rates of the pseudo-population.
    """

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
