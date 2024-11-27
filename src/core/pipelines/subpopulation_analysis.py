#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.pipelines.subpopulation_analysis` [module]

Classes
-------
`FormatPopulationData`

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

from core.pipelines.base_pipeline import Pipeline
from core.attributes.brain_info import Area, Training
from core.composites.exp_conditions import ExpCondition, ExpCondition
from core.attributes.exp_factors import Task, Attention, Category, Behavior
from core.composites.coordinate_set import CoordinateSet
from core.data_structures.trials_properties import TrialsProperties
from utils.io_data.loaders import Loader


class SubpopulationAnalysisCondition(ExpCondition):
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
