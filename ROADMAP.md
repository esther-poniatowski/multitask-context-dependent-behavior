# Feature Name: `recover-licks`

## Issues
- Main issue: #36
- Sub-Issue:  #

## Current Status
Currently working on the deployment script. Next step: test some components of the deployment script.
Started: `process_evp.py`, `process_evp.m`

## Tasks
- [X] Script to connect to the server.
- [ ] Deployment config to specify the mapping between the local directories and the server
  directories.
- [X] Deployment script to transfer files from the local machine to the server.
- [ ] Server scripts for setup (in `setup/`): create env, get matlab dependencies, adjust paths.
- [ ] Server package (in `src/`): logic to extract lick histograms in the server (see
  Notes).
- [ ] Server scripts (in `ops/`) to orchestrate the full process.


## Notes
MATLAB dependencies to install:
git clone https://platformneuro@bitbucket.org/abcng/baphy.git  for matlab

Procedure to extract data:
1. Locally gzip the *.evp.gz file
2. Call matlab script to load the data and save them in csv, with matlab -batch scriptname
3. Remove the locally unzipped folder

Content to extract from evp files:
[~,~,rAtot,ATrialIdxtot] = evpread('saf021b05_a_PTD.evp','auxchans',1);
% rAtot sampled at 1kHz
% rAtot is the signal over the entire session
% ATrialIdxtot is the index (in samples) of trials onsets

Example EVP path : /mnt/working2/esther/data4/MDdata/tan025b/tan025b01_p_FTC.evp.gz

## Analyses
Look to single trial lick signals.
If there are interruptions <50ms, 'complete' the signal.

## Resources
https://www.mathworks.com/help/matlab/ref/matlablinux.html

## Open Questions
Yves' suggestions :
1. Start matlab, cd in baphy folder ()
2. cmd: `baphy_set_path`
Is it necessary? Not clean process.
