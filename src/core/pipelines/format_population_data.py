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
from typing import Type, Dict, List

import numpy as np

from core.constants import N_FOLDS, N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.pipelines.base_pipeline import Pipeline
from core.processors.preprocess.count_trials import SampleSizer, TrialsCounter
from core.attributes.bio_info import Area, Training
from core.composites.exp_conditions import PipelineCondition
from core.composites.features import Features
from core.composites.containers import UnitsContainer, ExpCondContainer, Container
from core.composites.candidates import Candidates
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.coordinates.trials_coord import CoordPseudoTrialsIdx, CoordFolds
from core.builders.build_ensembles import EnsemblesBuilder
from core.builders.build_trial_coords import TrialCoordsBuilder
from core.builders.build_folds import FoldsBuilder
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
        units = Candidates(loaders_trial_prop.list_keys())
        # Load the trials' properties for each unit
        trials_props: UnitsContainer[TrialsProperties] = loaders_trial_prop.load()
        # Retrieve the coordinates of interest
        features = trials_props.apply(lambda tp: Features(**tp.get_coords_from_dim("features")))

        # Count the number of trials available for each unit in each condition
        counter = TrialsCounter(feat_by_unit=features.list_values())
        counts_actual = ExpCondContainer[np.ndarray].from_keys(
            keys=self.exp_conds.to_list(),
            fill_value=np.zeros(1),
            value_type=np.ndarray,
        )
        counts_actual.fill(counter.process)
        # Determine the number of trials to form in each condition based on the actual counts
        sizer = SampleSizer(k=self.k, n_min=self.n_min, thres_perc=self.thres_perc)
        counts_final = counts_actual.apply(sizer.process)
        # Exclude units with insufficient trials in any condition
        for cond, n_min in counts_final.items():
            units.filter_by_associated(values=counts_actual[cond], predicate=lambda x: x >= n_min)

        # Build ensembles (pseudo-populations)
        ens_size = len(units) if self.ensemble_size is None else self.ensemble_size
        builder_ens = EnsemblesBuilder(ens_size, self.n_ensembles_max)
        coord_units = builder_ens.build(units=units.to_list(), seed=0)
        # shape: (n_ensembles, ensemble_size)
        data_structure.set_coord("units", coord_units)
        # Split units into ensembles (make duplicate units independent from each other in distinct
        # ensembles): extract rows of `coord_units` (ensembles dimension)
        ensembles = Container(dict(enumerate(coord_units)), key_type=int, value_type=np.ndarray)

        # Initialize trial-related builders
        # Configure builders with shared parameters
        order_conditions = self.exp_conds.to_list()
        counts_by_condition = counts_final.to_dict()
        builder_folds = FoldsBuilder(self.k, order_conditions)
        builder_pseudo_trials = PseudoTrialsBuilder(self.k, counts_by_condition, order_conditions)
        # Build folds and pseudo-trials by ensemble
        for ens, ensemble in ensembles.items():
            # Build folds for each unit
            folds_labels = UnitsContainer(
                {
                    unit: builder_folds.build(features=features[unit], seed=u)
                    for u, unit in enumerate(ensemble)
                },
                value_type=CoordFolds,
            )

        # For each ensemble separately
        pseudo_trials_by_ensemble: List[CoordPseudoTrialsIdx] = []
        for ens, ensemble in enumerate(coord_units):
            feat_by_unit = features.list_values(ensemble)  # retrieve features of units in ensemble
            pseudo_trials = builder_pseudo_trials.build(feat_by_unit=feat_by_unit, seed=ens)
            pseudo_trials_by_ensemble.append(pseudo_trials)
        # Gather all pseudo-trials in a single coordinate
        coord_pseudo_trials = builder_pseudo_trials.gather_ensembles(pseudo_trials_by_ensemble)
        data_structure.set_coord("pseudo_trials_idx", coord_pseudo_trials)

        # Build trial-related coordinates
        builder_trials_coords = TrialCoordsBuilder(counts_by_condition, order_conditions)
        for name, coord_type in self.coords_trials.items():
            coord = builder_trials_coords.build(coord_type=coord_type)
            data_structure.set_coord(name, coord)

        # Create core data values (firing rates)
