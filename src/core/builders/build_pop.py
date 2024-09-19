#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.builders.build_pop` [module]

Classes
-------
:class:`PopulationBuilder`
"""

from typing import List, TypeAlias, Any, Tuple

import numpy as np

from core.data_structures.firing_rates import FiringRatesPop, FiringRatesUnit
from core.builders.base import DataBuilder
from core.processors.preprocess.stratify import Stratifier
from core.processors.preprocess.assign_ensembles import EnsembleAssigner
from core.processors.preprocess.assign_folds import FoldAssigner
from core.processors.preprocess.bootstrap import Bootstrapper


Strata: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]

Folds: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]

Ensembles: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]

PseudoTrials: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.float64]]


class PopulationBuilder(DataBuilder[FiringRatesPop]):
    """
    Build a :class:`FiringRatesPop` data structure.

    Inputs: List of firing rates (trials, time) for each unit in the population.
    Output: Firing rates of the pseudo-population in reconstructed pseudo-trials.

    Warning
    -------
    For data analysis:

    - Assign trials to folds *by unit*, within each ensemble. Justification: This increases the
      heterogeneity of the data for the case where two units are present in two identical ensembles.
      Indeed, since the folds of trials will be different,then the pairings of the trials are less
      likely match two identical trials from both units in both ensembles. This is possible if the
      seed varies for each unit when its folds are generated, which is the case if the unit does not
      have the same index in the population in both ensembles (which is likely to be the case since
      the units are shuffled when forming ensembles).
    - Use *stratified* assignment by condition (task, context, stimulus, error, fold, batch).
    """

    data_class = FiringRatesPop
    inputs = {"firing_rates_units": List[FiringRatesUnit]}

    def __init__(self):
        self.firing_rates_units = None
        self.n_units = None
        self.n_trials = None
        self.n_t = None
        self.time = None
        self.ensembles = None
        self.counts_by_ensemble = None
        self.strata_by_unit = None
        self.folds_by_unit = None
        self.pseudo_trials = None
        super().__init__()

    def build(self, *args, firing_rates_units=None, **kwargs) -> FiringRatesPop:
        """
        Implement the base class method.

        Parameters
        ----------
        firing_rates_units : List[FiringRatesUnit]
            List of firing rates for each unit in the population.
        """
        if firing_rates_units is None:
            firing_rates_units = []
        self.firing_rates_units = firing_rates_units
        self.n_units = len(firing_rates_units)
        return self.data_structure

    def assign_units_to_ensembles(self, ensemble_size: int):
        """
        Assign the units to ensembles.
        """
        assigner = EnsembleAssigner(ensemble_size=ensemble_size)
        assigner.process(n_units=self.n_units)
        ensembles = assigner.ensembles
        self.ensembles = ensembles

    def count_trials_by_ensemble(self):
        """
        Find the trial counts by ensemble.
        """
        trial_counts = []
        for units in self.ensembles:
            counts = np.array([self.firing_rates_units[u].n_trials for u in units])
            trial_counts.append(counts)
        self.counts_by_ensemble = trial_counts

    def determine_n_trials(self):
        """
        Determine the final number of pseudo-trials (homogeneous across ensembles).

        Notes
        -----
        Use the static method from the Bootstrapper class.
        """
        n_trials_by_ensemble = []
        for counts in self.counts_by_ensemble:
            n_trials = Bootstrapper.determine_n_pseudo(counts=counts)
            n_trials_by_ensemble.append(n_trials)
        n_trials = min(n_trials_by_ensemble)
        self.n_trials = n_trials

    def stratify_by_unit(self):
        """
        Stratify by unit.
        """
        strata_by_unit = []
        stratifier = Stratifier()
        for f_unit in self.firing_rates_units:
            features = None  # TODO: extract features from f_unit
            stratifier.process(features=features)
            strata = stratifier.strata
            strata_by_unit.append(strata)
        self.strata_by_unit = strata_by_unit

    def assign_trials_to_folds(self, k: int, strata_by_unit: List[Strata]):
        """
        Assign trials to folds by unit.
        """
        folds_by_unit = []
        assigner = FoldAssigner(k=k)
        for strata in strata_by_unit:
            assigner.process(strata=strata)
            folds = assigner.folds
            folds_by_unit.append(folds)
        self.folds_by_unit = folds_by_unit

    def bootstrap(self):
        """
        Bootstrap by ensemble and by unit. Stratify again before ?
        """
        pseudo_trials_by_ensemble = []
        bootstrapper = Bootstrapper(n_pseudo=self.n_trials)
        for ens, units in enumerate(self.ensembles):
            # count the trials for the units in the ensemble
            counts = self.counts_by_ensemble[ens]
            bootstrapper.process(counts=counts, seed=ens)  # seed by ensemble
            pseudo_trials = bootstrapper.pseudo_trials
            pseudo_trials_by_ensemble.append(pseudo_trials)
        self.pseudo_trials = pseudo_trials_by_ensemble

    def initialize_data_structure(self):
        """
        Initialize the array for the data and coordinates.

        Dimensions : ensemble, units, trials, time.
        Coordinates : all except positional info.
        """

    def pick_retained_trials(self):
        """
        Pick the retained trials and fill the array and coordinates.
        """

    def fill_trials(self):
        """
        Fill the array and coordinates with the retained trials.
        """

    def format_data_structure(self):
        """
        Format the data structure. Fill with coordinates and data.
        """
