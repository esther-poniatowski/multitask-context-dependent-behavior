#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.preprocess.hierarchical_bootstrap` [module]

Build pseudo-trials through an algorithm inspired from the "hierarchical bootstrap" method.

The goal is to associate the trials from each unit to reconstruct a pseudo-population.
This approach leverages multiple combinations of trials to augment the data set.
This overcomes the limitation on the resulting trials number to the minimum number of trials across
neurons.
*Prior* to the bootstrap process, for each unit independently, trials are gathered in sets:
- By task, context, stimulus.
- By folds
Hierarchical bootstrap can be applied on each set of trials.

Implementation
--------------
Input data is a list of trials' numbers for each unit.
Output data is an array of shape ``(n_units, n_pseudo)``, which contain the trial numbers selected
for each unit to form each pseudo-trial.
The final number of pseudo-trials is determined following specific rules (to be chosen) which might
depend on the minimum and the maximum number of trials across units.

The first function to implement is :func:`determine_pseudo_trial_number`.

Once the number of pseudo-trials is determined:

1. Trials are picked for each unit independently for future inclusion in the pseudo-trials.
2. Trials are combined across units to form actual pseudo-trials.
"""

import numpy as np
import numpy.typing as npt


def determine_pseudo_trial_number(counts: npt.ArrayLike) -> int:
    """
    Determine the number of pseudo-trials to generate.

    Rules #TODO

    Parameters
    ----------
    counts : npt.ArrayLike
        Counts of trials' numbers for each unit.

    Returns
    -------
    n_pseudo : int
        Number of pseudo-trials to generate.
    """
    return 0


def pick_trials(n: int, n_pseudo: int) -> np.ndarray:
    """
    Pick trials from a unit for future inclusion in the pseudo-trials.

    Implementation
    --------------
    The selection process aims to maximize the diversity of trials occurrences for each unit.
    Each trial might occur multiple times or not at all in the pseudo-trials depending on the number
    of trials available for the unit (``n``) and the required number pseudo-trials to generate (``n_pseudo``).
    Several cases are distinguished:

    - If ``n >= n_pseudo`` (*more* trials than the required number):
        Trials are randomly selected without replacement.
        It includes the case ``n == n_pseudo``, where all trials are selected once.
    - If ``n < n_pseudo`` (*less* trials than the required number):
        Each trial is selected at least ``n_pseudo // n`` times.
        To reach the required number of trials, ``n_pseudo % n`` remaining trials are randomly
        selected without replacement.

    Parameters
    ----------
    n : int
        Number of trials for the unit.
    n_pseudo : int
        Required number of pseudo-trials to generate.

    Returns
    -------
    trials : np.ndarray
        Trials selected for the unit.

    See Also
    --------
    :func:`np.random.choice`
    :func:`np.repeat`
    """
    if n >= n_pseudo:
        trials = np.random.choice(n, size=n_pseudo, replace=False)
    else:
        q = n_pseudo // n
        r = n_pseudo % n
        trials = np.repeat(np.arange(n), q)
        trials = np.concatenate((trials, np.random.choice(n, size=r, replace=False)))
    return trials


def combine_trials(trials: npt.NDArray) -> npt.NDArray:
    """
    Combine retained trials across units to form pseudo-trials.

    Implementation
    --------------
    To mitigate redundancies in trials' pairings, combinations are built by random shuffling of
    trials for each unit.
    This is a more straightforward approach than formulating an explicit algorithm to maximize the
    diversity of mutual trials combinations across units.

    Parameters
    ----------
    trials : npt.NDArray
        Trials selected for each unit.
        Shape: ``(n_units, n_pseudo)``.

    Returns
    -------
    pseudo : npt.NDArray
        Pseudo-trials formed by combining trials across units.
        Shape: ``(n_units, n_pseudo)`` (same as input).

    See Also
    --------
    :func:`np.random.shuffle`
    """
    for unit in trials:
        np.random.shuffle(unit)
    return trials
