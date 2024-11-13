#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.pipelines.subpopulation_analysis` [module]

Classes
-------
`SubpopulationAnalysis`

Notes
-----


"""


from core.pipelines.base_pipeline import Pipeline, PipelineFactory
from core.steps.concrete_steps import SelectUnits
from core.entities.bio_info import Area
from core.entities.exp_conditions import ExpCondition
from core.entities.exp_factors import Task, Attention, Stimulus, Behavior
from core.builders.build_ensembles import EnsemblesBuilder

from utils.io_data.loaders import LoaderCSVtoList


class SubpopulationAnalysis(Pipeline):
    """
    Pipeline for subpopulation analysis.

    Contributor: Hugo Tissot.

    Attributes
    ----------
	path_units : Path | str
		Path to the file containing the units in the population.
	path_excluded : Path | str
		Path to the file containing the units to exclude from the analysis.

    Notes
    -----
	General Attention
	^^^^^^^^^^^^^^^
    This analysis is one facet of a study which investigates the modularity of neuronal populations
    in the PFC during dual tasks.

    Hypothesis under investigation: The emergence of functional subpopulations in the PFC depends on
    the need for *sensory-motor remapping across tasks*.

    Three types of dual tasks are considered:

    - Dual Task 1 [current analysis]:
        - Two distinct stimuli sets (click rates in CLK / pure tones and noise in PTD)
        - Single motor output (Go/NoGo)
        -> No sensory-motor remapping for any stimulus
    - Dual Task 2:
        - Single stimulus set (Gabor Patches) with two classification boundaries (dimensions for
          classification: orientation or frequency)
        - Single motor output (Go/NoGo)
        -> Sensory-motor remapping for half of the stimuli (e.g. orientation Go and frequency NoGo
        or the reverse)
    - Dual Task 3:
        - Single stimulus set (Gabor Patches) with a single classification boundary (orientation ?)
        - Distinct motor outputs for the classification in two contexts (Go/NoGo, Left/Right)
        -> Sensory-motor remapping for all the stimuli

    Current Analysis
    ^^^^^^^^^^^^^^^^
    The current analysis focuses on the case of a dual task which does *not* involve sensory-motor
    remapping, since it involves two distinct stimuli sets (Dual Task 1).

    Goal: Probing the existence of functional subpopulations in the PFC area during the engagement
    in a dual task with two distinct sensory stimuli sets.

    Key Analyses
    ^^^^^^^^^^^^
    The presence of subpopulations is assessed in the space of neuronal "selectivity" to several
    task variables:

    - Stimulus category (Go/No-Go)
    - Choice (Lick/No-lick) (i.e. behavioral response)
    - Task (PTD/CLK)

    To identify subpopulations, two complementary methods are carried out:

    - Elliptical Projection Angle Index of Response Similarity (ePAIRS): Similarity-focused
      analysis, used to probe deviations from random mixed selectivity by measuring the angle
      between the selectivity vectors of pairs of neurons.
    - Gaussian Mixture Model (GMM): Model-based clustering analysis, used to estimate the number of
      latent subpopulations (each defined by a gaussian distribution of selectivity coefficients).

    Neuronal selectivity is evaluated by the coefficient of a linear regression model, which
    predicts the neuronal activity from the task variables. The full model includes 6 regressors:

    .. math::

        y = \\beta_{category-PTD} x_{category-PTD} + \\
            \\beta_{category-CLK} x_{category-CLK} + \\
            \\beta_{choice-PTD} x_{choice-PTD} + \\
            \\beta_{choice-CLK} x_{choice-CLK} + \\
            \\beta_{task} x_{task} + \\
            \\beta_{intercept}

	Limitation: In this model, the category and the sensory stimulus are confounded. Indeed, since
	only two stimuli are presented in each task, then the category is defined by the sensory
	stimulus.

    Improvements (ideas):

    - Discard the effect of the sensory stimulus: Replace the raw activity by the *difference*
      relative to the passive attentional state (where not category should be computed for the behavior).
    - Disambiguate the category from the sensory variable: Include in the model the neutral stimulus
      in CLK, which is perceptually identical to the reference in PTD.

    Implementation
    --------------
    Data Selection
    ^^^^^^^^^^^^^^
    Time window: For each trial, the neuronal response is averaged in the post-stimulus period (1
    second duration at the end of the trial, after the stimulus offset). This window encompasses the
    motor response and the shock delivery, and limits the effect of the sensory stimulus.

    Exclusions:

    - Passive attentional states (only include the trials in task engagement).
    - Neutral stimuli in the CLK task (not associated to any relevant category).

    Experimental conditions of interest:

    - Task: PTD, Category: Target (Tone), Behavior: Go
    - Task: PTD, Category: Target (Tone), Behavior: NoGo
    - Task: PTD, Category: Reference (Noise), Behavior: Go (NoGo does not exist)
    - Task: CLK, Category: Target (Click Rate 1), Behavior: Go
    - Task: CLK, Category: Target (Click Rate 1), Behavior: NoGo
    - Task: CLK, Category: Target (Click Rate 2), Behavior: Go (NoGo does not exist)

    Preprocessing steps
    ^^^^^^^^^^^^^^^^^^^
    - Reconstruct pseudo-trials for each experimental condition.
    - Define the regressor matrix X and normalize the regressors.
    - Z-score the activity of each neuron across all the pseudo-trials.
    - Denoise with Principal Component Analysis (PCA): Fit the PCA on the means intra-condition,
      keep the first components which explain 90% of the variance, invert the transformation to
      obtain the regressand matrix Y (to predict).

    Analysis Steps
	^^^^^^^^^^^^^^
    - Fit the linear regression model and retrieve the coefficients.
    - Assess the model with cross-validation ?
    - Compute the ePAIRS distribution across the population of neurons.
    - Fit a GMM on the distribution of the selectivity coefficients.
    - Estimate the number of subpopulations by the log-likelihood increase for each additional
      cluster

    Data Structures Milestones
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    - TrialsProperties
    - PseudoTrialsSelection
    - Regressors (matrix X)
    - FiringRatesPopulation (matrix Y, both before and after z-scoring and denoising)
    - SelectivityCoefficients (matrix B)
    - EPAIRSModel
    - GMMModel

    """

    REQUIRED_PATHS = frozenset({"path_units", "path_excluded"})

    def execute(self, **kwargs) -> None:
        """
        Implement the abstract method from the base class `Pipeline`.
        """
        # Identify PFC neurons
        all_units = LoaderCSVtoList(self.path_units).load()  # type: ignore[attr-defined] # pylint: disable=no-member
        excluded = LoaderCSVtoList(self.path_excluded).load()  # type: ignore[attr-defined] # pylint: disable=no-member
        selector = SelectUnits(all_units, excluded)
        units = selector.execute()

        # Build ensembles and pseudo-populations
        ensemble_size = len(units)  # all units in the population
        n_ensembles_max = 1  # only one pseudo-population
        builder = EnsemblesBuilder(ensemble_size=ensemble_size, n_ensembles_max=n_ensembles_max)
        coord_units = builder.build(units=units, seed=0)

        # Build pseudo-trials
        # Select the trials in the 4 conditions of interest
        attn = Attention("a")  # always active (task engagement)
        conditions = ExpCondition.generate_conditions()
        # Set the number of pseudo-trials to form in each condition
