# Feature Name: `select-trials`

## Issues
- Main issue: #48
- Sub-Issue:  #60

## Current Status

## To Do
- [ ]


## Next steps


## Open Questions


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
