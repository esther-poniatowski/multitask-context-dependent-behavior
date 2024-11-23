# Feature Name: `select-trials`

## Issues
- Main issue: #48
- Sub-Issue:  #60

## Current Status

## To Do
- [ ] Create a Config object for the different pipelines and analyses.
- [ ] Find a way to label the experimental conditions for access with a descriptive label.
- [ ] Add a functionality in pipelines to flexibly select steps to run and restart on checkpoints
- [ ] Implement data structure methods required by the builders: `iter_slots` in the session's metadata.
- [ ] Update tests for the concrete processors, coordinates, entities...
- [ ] Write tests for the builders, z-score.
- [ ] Update documentations in `__init__.py` files.
- [ ] Refactor SpikesAligner to reduce conditionals: Create Task/Stimulus Subclasses PTDAligner and
  CLKAligner under a shared BaseAligner interface.

## Next steps
- Implement the data structure for single units firing rates and the corresponding builder.
- Add a `collect` method to the builder base class to gather all the required data for building the
  final object.
- Implement a processor `TrialOrganizer` to extract the trials from the sessions' blocks.
- Implement the concrete data structure class for storing the neuron-specific properties of the
  trials and their inclusion/exclusion in the analysis.
- Implement a `TrialSelector` class

## Open Questions

What questions should be addressed ?

I have to design a workflow which carries out a full processing of the data. It takes as input:
- The different analyses that should be performed.
- The areas, units, contexts or tasks that should be considered (depending on the analysis).


Understand why the steps are coupled in the builder and find how to reduce this coupling.

So I am going to create a Pipeline interface and then different pipelines for different analyses via
a Pipeline Factory.

The Steps will be analogous to the Directors of the builders (when they are aimed to build data).
