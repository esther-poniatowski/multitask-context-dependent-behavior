#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.pipelines.subpopulation_analysis` [module]

Classes
-------
`SubpopulationAnalysis`

Notes
-----
Preprocessing steps:

- Reconstruct pseudo-trials for each experimental condition.
- Define the regressor matrix X and normalize the regressors.
- Z-score the activity of each neuron across all the pseudo-trials.
- Denoise with Principal Component Analysis (PCA): Fit the PCA on the means intra-condition, keep
  the first components which explain 90% of the variance, invert the transformation to obtain the
  regressand matrix Y (to predict).

Analysis Steps:

- Fit the linear regression model and retrieve the coefficients.
- Assess the model with cross-validation ?
- Compute the ePAIRS distribution across the population of neurons.
- Fit a GMM on the distribution of the selectivity coefficients.
- Estimate the number of subpopulations by the log-likelihood increase for each additional cluster

Data Structures Milestones:

- TrialsProperties
- PseudoTrialsSelection
- Regressors (matrix X)
- FiringRatesPopulation (matrix Y, both before and after z-scoring and denoising)
- SelectivityCoefficients (matrix B)
- EPAIRSModel
- GMMModel


"""
from types import MappingProxyType
from typing import List, Type, Dict

from core.pipelines.base_pipeline import Pipeline
from core.processors.preprocess.exclude import Excluder
from core.entities.bio_info import Area, Training
from core.entities.exp_conditions import PipelineCondition
from core.entities.exp_factors import Task, Attention, Category, Behavior
from core.coordinates.coord_manager import CoordManager
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.builders.build_ensembles import EnsemblesBuilder
from core.builders.build_trial_coords import TrialCoordsBuilder
from core.data_structures.firing_rates_pop import FiringRatesPop
from core.data_structures.trials_properties import TrialsProperties
from utils.io_data.loaders import Loader
from utils.storage_rulers.base_path_ruler import PathRuler


from utils.io_data.loaders import LoaderCSVtoList


class SubpopulationAnalysisCondition(PipelineCondition):
    """
      Experimental condition tailored for the `SubpopulationAnalysis` pipeline:

      - Task: PTD or CLK
      - Attention: Always active (a)
      - Category: R or T
      - Behavior: Go or NoGo

    - Task: PTD, Category: Target (Tone), Behavior: Go
    - Task: PTD, Category: Target (Tone), Behavior: NoGo
    - Task: PTD, Category: Reference (Noise), Behavior: Go (NoGo does not exist)
    - Task: CLK, Category: Target (Click Rate 1), Behavior: Go
    - Task: CLK, Category: Target (Click Rate 1), Behavior: NoGo
    - Task: CLK, Category: Target (Click Rate 2), Behavior: Go (NoGo does not exist)

    """

    REQUIRED_FACTORS = MappingProxyType(
        {
            "task": frozenset([Task("PTD"), Task("CLK")]),
            "attention": frozenset([Attention("a")]),  # Fixed value
            "category": frozenset([Category("R"), Category("T")]),
            "behavior": frozenset([Behavior("Go"), Behavior("NoGo")]),
        }
    )


class SubpopulationAnalysis(Pipeline):
    """
    Pipeline to format population data.

    Attributes
    ----------
        path_units : Path | str
                Path to the file containing the units in the population.
        path_excluded : Path | str
                Path to the file containing the units to exclude from the analysis.
    """

    PATH_INPUTS = frozenset(["path_units", "path_excluded", "path_trial_prop"])
    PATH_OUTPUTS = frozenset()
    LOADERS = frozenset(["loader_units", "loader_excluded", "loader_trial_prop"])
    SAVERS = frozenset()

    def __init__(
        self,
        area: Area,
        training: Training,
        path_units: PathRuler,
        path_excluded: PathRuler,
        path_trial_prop: PathRuler,
        exp_cond_type: PipelineCondition,
        ensemble_size: int | None = None,  # all units in the population
        n_ensembles_max: int = 1,  # only one pseudo-population
        coords_trials: Dict[str, Type[CoordExpFactor]] = {},
        loader_units: Type[Loader] = LoaderCSVtoList,
        loader_excluded: Type[Loader] = LoaderCSVtoList,
        loader_trial_prop: Type[Loader] = LoaderCSVtoList,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.area = area
        self.training = training
        self.exp_cond_type = exp_cond_type
        self.ensemble_size = ensemble_size
        self.n_ensembles_max = n_ensembles_max
        self.coords_trials = coords_trials
        self.loader_units = loader_units
        self.loader_excluded = loader_excluded
        self.loader_trial_prop = loader_trial_prop
        self.path_units = path_units
        self.path_excluded = path_excluded
        self.path_trial_prop = path_trial_prop

    def execute(self, **kwargs) -> None:
        """
        Implement the abstract method from the base class `Pipeline`.
        """
        # Initialize the data structure
        data_structure = FiringRatesPop(area=self.area, training=self.training)

        # Identify neurons in the area of interest
        all_units = self.loader_units(self.path_units.get_path()).load()
        excluded = self.loader_excluded(self.path_excluded.get_path()).load()
        units = Excluder().process(candidates=all_units, excluded=excluded)
        # Retrieve each unit's file containing its trials properties
        trials_props: List[TrialsProperties] = [self.loader_trial_prop(self.path_trial_prop(unit).get_path()).load() for unit in units]  # type: ignore
        assert all([isinstance(tp, TrialsProperties) for tp in trials_props])

        # Build pseudo-trials
        # Define the conditions of interest
        exp_conds = self.exp_cond_type.generate()  # ExpConditionUnion = list of ExpCondition
        # Count the number of trials available for each unit in the population
        counts = {cond: {unit: 0 for unit in units} for cond in exp_conds}
        for unit, tp in zip(units, trials_props):
            coords = CoordManager(**tp.get_coords_from_dim("trials"))
            for cond in exp_conds:
                counts[cond][unit] = coords.count(cond)

        # Build the trial-related coordinates
        n_by_cond = {cond: 0 for cond in exp_conds}  # TODO
        builder_trials_coords = TrialCoordsBuilder(n_by_cond=n_by_cond)
        for name, coord_type in self.coords_trials.items():
            coord = builder_trials_coords.build(coord_type=coord_type)
            data_structure.set_coord(name, coord)

        # Build ensembles (pseudo-populations)
        ens_size = len(units) if self.ensemble_size is None else self.ensemble_size
        builder_ens = EnsemblesBuilder(ensemble_size=ens_size, n_ensembles_max=self.n_ensembles_max)
        coord_units = builder_ens.build(units=units, seed=0)
        data_structure.set_coord("units", coord_units)


# The relevant file has the format of a data structure and contains the experimental factors for
# each trial (slot). It is the result of parsing the .m files for each session, but is now unit
# specific to allow specification of excluded trials. It is located in each unit's directory. To
# retrieve it, use the dill loader with a path ruler which takes the unit name as an argument.


# Set the number of pseudo-trials to form in each condition
