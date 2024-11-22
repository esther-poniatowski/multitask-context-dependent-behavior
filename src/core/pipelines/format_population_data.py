#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.pipelines.format_population_data` [module]

Classes
-------
`FormatPopulationData`


# The relevant file has the format of a data structure and contains the experimental factors for
# each trial (slot). It is the result of parsing the .m files for each session, but is now unit
# specific to allow specification of excluded trials. It is located in each unit's directory. To
# retrieve it, use the dill loader with a path ruler which takes the unit name as an argument.

"""
from types import MappingProxyType
from typing import Type, Dict

import numpy as np

from core.constants import N_FOLDS, N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.pipelines.base_pipeline import Pipeline
from core.processors.preprocess.exclude import Excluder
from core.processors.preprocess.count_trials import SampleSizer, TrialsCounter
from core.entities.bio_info import Area, Training, Unit
from core.composites.exp_conditions import PipelineCondition, ExpCondition
from core.entities.exp_factors import Task, Attention, Category, Behavior
from core.composites.features import Features
from core.composites.containers import UnitsContainer, ExpCondContainer
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.builders.build_ensembles import EnsemblesBuilder
from core.builders.build_trial_coords import TrialCoordsBuilder
from core.builders.build_pseudo_trials import PseudoTrialsBuilder
from core.data_structures.firing_rates_pop import FiringRatesPop
from core.data_structures.trials_properties import TrialsProperties
from utils.io_data.loaders import Loader


class FormatPopulationData(Pipeline):
    """
    Pipeline to format population data.

    Attributes
    ----------

    """

    PATH_INPUTS = frozenset(["path_units", "path_excluded", "path_trial_prop"])
    PATH_OUTPUTS = frozenset()
    LOADERS = frozenset(["loader_units", "loader_excluded", "loader_trial_prop"])
    SAVERS = frozenset()

    def __init__(
        self,
        exp_cond_type: PipelineCondition,
        ensemble_size: int | None = None,  # all units in the population
        n_ensembles_max: int = 1,  # only one pseudo-population
        k: int = N_FOLDS,
        n_min: int = N_TRIALS_MIN,
        thres_perc: float = BOOTSTRAP_THRES_PERC,
        coords_trials: Dict[str, Type[CoordExpFactor]] | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Define the experimental conditions of interest
        self.exp_cond_type = exp_cond_type
        self.exp_conds = self.exp_cond_type.generate()  # ExpConditionUnion = list of ExpCondition
        #
        self.ensemble_size = ensemble_size
        self.n_ensembles_max = n_ensembles_max
        self.coords_trials = coords_trials if coords_trials is not None else {}
        self.k = k
        self.n_min = n_min
        self.thres_perc = thres_perc

    def execute(
        self,
        area: Area = Area("PFC"),
        training: Training = Training(False),
        loaders_trial_prop: UnitsContainer[Loader] | None = None,
        **kwargs
    ) -> None:
        """
        Implement the abstract method from the base class `Pipeline`.

        Arguments
        ---------
        area : Area
            Brain area of interest.
        training : Training
            Training status of the animals.
        loaders_trial_prop : UnitsContainer[Loader]
            Loaders for the files containing the trial properties of each unit in the population.
        """
        assert loaders_trial_prop is not None

        # Initialize the data structure
        data_structure = FiringRatesPop(area=area, training=training)

        # Retrieve units
        units = loaders_trial_prop.units
        # Load the trials' properties for each unit
        trials_props: UnitsContainer[TrialsProperties] = loaders_trial_prop.load()
        # Retrieve the coordinates of interest
        features = trials_props.apply(lambda tp: Features(**tp.get_coords_from_dim("features")))

        # Count the number of trials available for each unit in each condition
        counter = TrialsCounter(feat_by_unit=features.fetch())
        counts_actual = ExpCondContainer(
            {cond: counter.process(cond) for cond in self.exp_conds}, value_type=np.ndarray
        )
        # Determine the number of trials to form in each condition based on the actual counts
        sizer = SampleSizer(k=self.k, n_min=self.n_min, thres_perc=self.thres_perc)
        counts_final = counts_actual.apply(sizer.process)
        # Exclude units with insufficient trials # TODO
        for cond, n_min in counts_final.items():
            units = list(Excluder.exclude_from_counts(counts_actual[cond], n_min))

        # Build ensembles (pseudo-populations)
        ens_size = len(units) if self.ensemble_size is None else self.ensemble_size
        builder_ens = EnsemblesBuilder(ens_size, self.n_ensembles_max)
        coord_units = builder_ens.build(units=units, seed=0)  # shape: (n_ensembles, ensemble_size)
        data_structure.set_coord("units", coord_units)

        # Build pseudo-trials
        order_conditions = self.exp_conds.get()
        builder_pseudo_trials = PseudoTrialsBuilder(self.k, counts_final, order_conditions)
        # For each ensemble separately
        pseudo_trials_by_ensemble = []
        for ens, ensemble in enumerate(coord_units):  # rows of `coord_units` = ensembles
            coords_ens = ensemble.iter_through(feat_by_unit, units)
            pseudo_trials = builder_pseudo_trials.build(feat_by_unit=coords_ens, seed=ens)
            pseudo_trials_by_ensemble.append(pseudo_trials)
        # Gather all pseudo-trials in a single coordinate
        coord_pseudo_trials = builder_pseudo_trials.gather_ensembles(pseudo_trials_by_ensemble)

        # Build the trial-related coordinates
        builder_trials_coords = TrialCoordsBuilder(counts_by_condition, order_conditions)
        for name, coord_type in self.coords_trials.items():
            coord = builder_trials_coords.build(coord_type=coord_type)
            data_structure.set_coord(name, coord)
