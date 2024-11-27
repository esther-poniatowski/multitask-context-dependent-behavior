#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.pipelines.format_population_data` [module]

Classes
-------
`FormatPopulationData`




"""
from typing import Type, Dict, List, Any
from dataclasses import dataclass, field

import numpy as np

from core.constants import N_FOLDS, N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.pipelines.base_pipeline import Pipeline, PipelineConfig, PipelineInputs
from core.processors.preprocess.count_samples import SampleSizer, TrialsCounter
from core.attributes.brain_info import Area, Training
from core.composites.exp_conditions import ExpCondition
from core.composites.coordinate_set import CoordinateSet
from core.composites.base_container import Container
from core.composites.containers_fixed import UnitsContainer, ExpCondContainer
from core.composites.candidates import Candidates
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.coordinates.trial_analysis_label_coord import CoordPseudoTrialsIdx, CoordFolds
from core.builders.build_ensembles import EnsemblesBuilder
from core.builders.build_trial_coords import TrialCoordsBuilder
from core.builders.build_folds import FoldsBuilder
from core.builders.build_pseudo_trials import PseudoTrialsBuilder
from core.data_structures.firing_rates_pop import FiringRatesPop
from core.data_structures.trials_properties import TrialsProperties


@dataclass
class FormatPopulationDataConfig(PipelineConfig):
    """
    Configuration for the pipeline to format population data.

    Attributes
    ----------
    exp_cond_type : Type[ExpCondition]
        Type of experimental condition to consider.
    ensemble_size : int | None
        Number of units in each ensemble. Default: all units in the population.
    n_ensembles_max : int
        Maximum number of pseudo-populations to form. Default: single pseudo-population.
    n_folds : int
        Number of folds for cross-validation. Default: 3.
    n_min : int
        Minimum number of required trials for a unit to be included in the analysis, in each
        experimental condition of interest and fold. Default: 5.
    thres_perc : float
        Threshold percentage of the number of trials to consider the bootstrap method. Default: 0.3.
    coords_trials : Dict[str, Type[CoordExpFactor]]
        Coordinates to build for the trial analysis.

    See Also
    --------
    `dataclasses.dataclass`
    """

    exp_cond_type: ExpCondition
    ensemble_size: int | None = None  # default: all units in the population
    n_ensembles_max: int = 1  # default: single pseudo-population
    n_folds: int = N_FOLDS
    n_min: int = N_TRIALS_MIN
    thres_perc: float = BOOTSTRAP_THRES_PERC
    coords_trials: Dict[str, Type[CoordExpFactor]] = field(default_factory=dict)


@dataclass
class FormatPopulationDataInputs(PipelineInputs):
    """
    Inputs for the pipeline to format population data.

    Attributes
    ----------
    area : Area
        Brain area of interest.
    training : Training
        Training status of the animals.
    units : Candidates
        Data structure containing the units of the population of interest, before filtering for
        sufficient number of trials in each condition.
    trials_properties: UnitsContainer[TrialsProperties]
        Data structures containing the trial properties of each unit in the population
        (experimental factors for each trial "slot"). It is the result of parsing the ``.m``
        files for each session, but is now unit-specific.

    See Also
    --------
    `dataclasses.dataclass`
    """

    area: Area
    training: Training
    units: Candidates
    trials_properties: UnitsContainer[TrialsProperties]


class FormatPopulationData(Pipeline[FormatPopulationDataConfig, FormatPopulationDataInputs]):
    """
    Pipeline to format population data.
    """

    def __init__(self, config: FormatPopulationDataConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.exp_conds = self.config.exp_cond_type.generate()  # ExpConditionUnion

    def execute(self, inputs: FormatPopulationDataInputs, **kwargs: Any) -> None:
        """
        Implement the abstract method from the base class `Pipeline`.
        """
        area = inputs.area
        training = inputs.training
        units = inputs.units
        trials_properties = inputs.trials_properties

        # Initialize the data structure
        data_structure = FiringRatesPop(area=area, training=training)

        # Retrieve the coordinates of interest
        features = trials_properties.apply(
            lambda tp: CoordinateSet(**tp.get_coords_from_dim("trials"))
        )

        # Count the number of trials available for each unit in each condition
        counter = TrialsCounter(features_by_unit=features.list_values())
        counts_actual = ExpCondContainer[np.ndarray].from_keys(
            keys=self.exp_conds.to_list(),
            fill_value=np.zeros(1),
            value_type=np.ndarray,
        )
        counts_actual.fill(counter.process)
        # Determine the number of trials to form in each condition based on the actual counts
        sizer = SampleSizer(
            n_folds=self.config.n_folds, n_min=self.config.n_min, thres_perc=self.config.thres_perc
        )
        counts_final = counts_actual.apply(sizer.process)
        # Exclude units with insufficient trials in any condition
        for cond, n_min in counts_final.items():
            units.filter_by_associated(values=counts_actual[cond], predicate=lambda x: x >= n_min)

        # Build ensembles (pseudo-populations)
        ens_size = len(units) if self.config.ensemble_size is None else self.config.ensemble_size
        builder_ens = EnsemblesBuilder(ens_size, self.config.n_ensembles_max)
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
        builder_folds = FoldsBuilder(self.config.n_folds, order_conditions)
        builder_pseudo_trials = PseudoTrialsBuilder(
            self.config.n_folds, counts_by_condition, order_conditions
        )
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
            features_by_unit = features.list_values(
                ensemble
            )  # retrieve features of units in ensemble
            pseudo_trials = builder_pseudo_trials.build(features_by_unit=features_by_unit, seed=ens)
            pseudo_trials_by_ensemble.append(pseudo_trials)
        # Gather all pseudo-trials in a single coordinate
        coord_pseudo_trials = builder_pseudo_trials.gather_ensembles(pseudo_trials_by_ensemble)
        data_structure.set_coord("pseudo_trials_idx", coord_pseudo_trials)

        # Build trial-related coordinates
        builder_trials_coords = TrialCoordsBuilder(counts_by_condition, order_conditions)
        for name, coord_type in self.config.coords_trials.items():
            coord = builder_trials_coords.build(coord_type=coord_type)
            data_structure.set_coord(name, coord)

        # Create core data values (firing rates)
