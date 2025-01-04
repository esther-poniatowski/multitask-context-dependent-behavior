Geometrical Analysis - Baseline Activity
========================================

.. _geom_baseline_analysis:

General Context
---------------
This analysis is one facet of the study of the geometry of neuronal activity in a hierarchy of brain
areas during dual tasks with attentional modulations.

Four brain areas are considered, from the auditory sensory to the associative cortex: A1, dPEG, VPr,
and PFC.

The geometrical perspective on neuronal activity consists in assimilating neuronal patters to
vectors in a high-dimensional space where each dimension corresponds to one neuron. Thus, assessing
the "geometry" of the neuronal activity involves describing the structural distribution of the
patterns within this multidimensional space, in response to distinct experimental conditions. The
ultimate goal is to elucidate how distinct experimental variables manifest in the neuronal
representations at the population-level.

Specifically, the analysis aims to determine whether the neuronal activity can be described by
linear or non-linear functions of experimental variables. In the linear case, each variable might
correspond to a specific axis within the neuronal space. Alternatively, in the non-linear space, the
interaction of variables could generate complex transformations—such as rotations and scaling—of
neuronal responses.

Current Analysis
----------------
The current analysis focuses on the baseline activity ("spontaneous" activity, in the absence of any
stimulus presentation).

Goal: Establishing the geometry of the baseline activity in each brain area, in a dual task (two
stimuli sets) with attentional modulations (active task engagement vs. passive perception).

Hypotheses under investigation (for each brain area):

- Modulation by the task (stimulus set):
    - H1: No effect in both attentional states (passive and active).
    - H2: Effect in the active attentional state only.
    - H3: Effect in both attentional states, similar.
    - H4: Effect in both attentional states, different.
- Modulation by the attentional state (engagement in task performance):
    - H5: No effect in both tasks.
    - H6: Effect in both tasks, similar.
    - H7: Effect in both tasks, different

Key Analyses
------------
The "geometry" of the neuronal activity is assessed by linear decoding of the two experimental
variables: task and attentional state.

Data Selection
--------------
Time window: Baseline activity, before the stimulus onset.

Exclusions: Error trials

Experimental conditions of interest:

- Task: PTD, Attention: Active, Category: Go
- Task: PTD, Attention: Active, Category: No-Go
- Task: PTD, Attention: Passive, Category: Go
- Task: PTD, Attention: Passive, Category: No-Go
- Task: CLK, Attention: Active, Category: Go
- Task: CLK, Attention: Active, Category: No-Go
- Task: CLK, Attention: Passive, Category: Go
- Task: CLK, Attention: Passive, Category: No-Go
