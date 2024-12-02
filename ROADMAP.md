# Feature Name: `select-trials`

## Issues
- Main issue: #48
- Sub-Issue:  #60

## Current Status

## To Do
- In coordinates, add interactions with attributes. Other subclasses are different data values.
- Distinguish between data components and data structures.
- Distinguish between builders and factories
- Implement TypeDict for coordinate mappings in DataStructure
- Implement a processor to gather folds, ensembles...
- [ ] Refactor SpikesAligner to reduce conditionals: Create Task/Stimulus Subclasses PTDAligner and
  CLKAligner under a shared BaseAligner interface.
- [ ] Add a functionality in pipelines to flexibly select steps to run and restart on checkpoints
- [ ] Update tests.
- [ ] Update documentations in `__init__.py` files.
- [ ] Start back geom-baseline analysis and subpopulation analysis once the interface is fixed.

## Next steps
- Add a `collect` method to the builder base class to gather all the required data for building the
  final object.
- Implement a processor `TrialOrganizer` to extract the trials from the sessions' blocks.
 in the analysis.
- Implement a `TrialSelector` class

## Open Questions

Improvements:
- Break down the execute method into smaller, more focused methods
- Use a factory method for creating ExpCondition instances
- Use a dataclass for attributes that define the pipeline configuration. This makes the code easier
  to read and ensures type validation.
- Configuration object
- Use dependency injection for external dependencies like ExpCondition, Candidates, and UnitsContainer.

## Saveguard from `subpopulation_analysis`

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
