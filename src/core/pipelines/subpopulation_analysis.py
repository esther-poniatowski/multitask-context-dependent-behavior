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

# The relevant file has the format of a data structure and contains the experimental factors for
# each trial (slot). It is the result of parsing the .m files for each session, but is now unit
# specific to allow specification of excluded trials. It is located in each unit's directory. To
# retrieve it, use the dill loader with a path ruler which takes the unit name as an argument.

"""
from types import MappingProxyType
from typing import List, Type, Dict

from core.constants import N_FOLDS, N_TRIALS_MIN, BOOTSTRAP_THRES_PERC
from core.pipelines.base_pipeline import Pipeline
from core.processors.preprocess.exclude import Excluder
from core.processors.preprocess.count_trials import SampleSizer, TrialsCounter
from core.entities.bio_info import Area, Training
from core.entities.exp_conditions import PipelineCondition, ExpCondition
from core.entities.exp_factors import Task, Attention, Category, Behavior
from core.coordinates.coord_manager import CoordManager
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.builders.build_ensembles import EnsemblesBuilder
from core.builders.build_trial_coords import TrialCoordsBuilder
from core.data_structures.firing_rates_pop import FiringRatesPop
from core.data_structures.trials_properties import TrialsProperties
from utils.io_data.loaders import Loader


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
        loader_units: Loader | None = None,
        loader_excluded: Loader | None = None,
        loaders_trial_prop: List[Loader] | None = None,
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
        loader_units : Loader
            Loader for the file containing the units in the population.
        loader_excluded : Loader
            Loader for the file containing the units to exclude from the analysis.
        loaders_trial_prop : List[Loader]
            Loaders for the files containing the trial properties of each unit in the population.
        """
        assert loader_units is not None
        assert loader_excluded is not None
        assert loaders_trial_prop is not None

        # Initialize the data structure
        data_structure = FiringRatesPop(area=area, training=training)

        # Identify neurons in the area of interest
        all_units = loader_units.load()
        excluded = loader_excluded.load()
        units = Excluder.exclude_by_difference(candidates=all_units, intruders=excluded)
        # Retrieve each unit's file containing its trials properties
        trials_props: List[TrialsProperties] = [loader.load() for loader in loaders_trial_prop]
        assert all([isinstance(tp, TrialsProperties) for tp in trials_props])
        # Retrieve the coordinates of interest
        coords_by_unit = [CoordManager(**tp.get_coords_from_dim("trials")) for tp in trials_props]

        # Determine the number of trials to form in each condition
        # Count the number of trials available for each unit in the population
        counter = TrialsCounter(coords_by_unit=coords_by_unit)
        counts_by_cond = {cond: counter.process(exp_cond=cond) for cond in self.exp_conds}
        n_by_cond: Dict[ExpCondition, int] = {
            cond: SampleSizer(counts).process(
                k=self.k, n_min=self.n_min, thres_perc=self.thres_perc
            )
            for cond, counts in counts_by_cond.items()
        }

        # Build pseudo-trials

        # Build the trial-related coordinates
        builder_trials_coords = TrialCoordsBuilder(n_by_cond=n_by_cond)
        for name, coord_type in self.coords_trials.items():
            coord = builder_trials_coords.build(coord_type=coord_type)
            data_structure.set_coord(name, coord)

        # Build ensembles (pseudo-populations)
        ens_size = len(units) if self.ensemble_size is None else self.ensemble_size
        builder_ens = EnsemblesBuilder(ensemble_size=ens_size, n_ensembles_max=self.n_ensembles_max)
        coord_units = builder_ens.build(units=units, seed=0)
        data_structure.set_coord("units", coord_units)
