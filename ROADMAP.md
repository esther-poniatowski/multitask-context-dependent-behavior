# ROADMAP

## Data ETL

- Correct `mount_data.sh` to allow optimal path specification to the credential file.
- Document the template file and how to use it.
- Remove any password present in the repository.
- Recover the credentials for each server.
- Correct links in rst files.

LSP computer
$(USER_LSP)@129.199.80.205:~/Documents/esther-transfer-MDdata

BiggerGuy
esther@129.199.80.162:/mnt/working2/esther

Explain:

- What are the servers involved in the ETL process and their respective roles.
- The file systems in the Maryland servers, the Bigger Guy, and the final one on the NAS.
- How to transfer data from the Maryland to the NAS, and the BigguerGuy to the NAS.
- The raw data structures and their content, and the exported formats.
- How to extract spikes, lick rates, and session information.
- How to transfer those files to the server performing the analyses?

Subsidiary questions:

- How to name the Maryland server? And the laboratory?
- How to name the server which performs the analyses?

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
