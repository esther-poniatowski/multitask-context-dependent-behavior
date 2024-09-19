# Feature Name: `select-trials`

## Issues
- Main issue: #48
- Sub-Issue:  #60

## Current Status
- Add a `collect` method to the builder base class to gather all the required data for building the
  final object.
- Implement a processor `TrialOrganizer` to extract the trials from the sessions' blocks.
- Turn firing rate utilities to class inheriting from Processor.
- Gather the trials of all the sessions for one recording site in `TrialsProperties`.
- Implement the concrete data structure class for storing the neuron-specific properties of the
  trials and their inclusion/exclusion in the analysis.
- Recover experimental info dataframes to convert it in a new data structure.

## Notes


## Open Questions
