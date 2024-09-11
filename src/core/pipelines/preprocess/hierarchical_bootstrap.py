#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.pipelines.preprocess.hierarchical_bootstrap` [module]

Build pseudo-trials through an algorithm inspired from the "hierarchical bootstrap" method.

Reconstructing a pseudo-population of units consists in associating their respective trials which
were not recorded simultaneously during the experiment.
To augment the resulting data set, the current approach inspires from the "hierarchical bootstrap"
method, which leverages multiple combinations of trials. Thereby, the data set size is less
limited by the minimum minimum number of trials across neurons.

Warning
-------
*Prior* to the bootstrap process, for each unit independently, trials should be gathered in sets:
- By task, context, stimulus
- By folds
Hierarchical bootstrap can be applied on each set of trials.

Implementation
--------------
Input data is a list of trials' counts for each unit.
Output data is an array of shape ``(n_units, n_pseudo)``, which contain the trial numbers selected
for each unit to form each pseudo-trial.
Processing steps:
1. Determine the number of pseudo-trials to generate.
2. Pick trials for each unit independently for future inclusion in the data set.
3. Combine trials across units to form actual pseudo-trials.
"""
import numpy as np
import numpy.typing as npt

from core.constants import N_PSEUDO_MIN, ALPHA_BOOTSTRAP


def determine_pseudo_trial_number(
    counts: npt.NDArray[np.int_],
    n_pseudo_min=N_PSEUDO_MIN,
    alpha=ALPHA_BOOTSTRAP,
) -> int:
    """
    Determine the number of pseudo-trials to generate based on the counts of trials for each unit.

    Rules:
    The number of pseudo-trials is based on:
    - A balance between the minimum and maximum number of trials across units, to avoid
      over-representing units with numerous trials and under-representing those with fewer trials.
    - A Variability Factor (alpha) to adjust the number of trials to achieve sufficient diversity.
      For example, setting ``alpha = 0.5`` is equivalent to choosing the average of the minimum and
      maximum number of trials across units, which promotes a moderate level of variability.

    Parameters
    ----------
    counts : npt.NDArray[np.int_]
        Counts of trials' numbers for each unit.
    n_pseudo_min : int, optional
        Minimum number of pseudo-trials in a condition and fold.
    alpha : float, default=ALPHA_BOOTSTRAP
        Variability factor to adjust the number of pseudo-trials.

    Returns
    -------
    n_pseudo : int
        Number of pseudo-trials to generate.
    """
    n_min = np.min(counts)
    n_max = np.max(counts)
    n_pseudo = int(alpha * (n_min + n_max))
    n_pseudo = max(n_pseudo, n_pseudo_min)  # at least `n_pseudo_min`
    return n_pseudo


def pick_trials(n: int, n_pseudo: int) -> np.ndarray:
    """
    Pick trials for a single unit for future inclusion in the pseudo-trials.

    Implementation
    --------------
    The selection process aims to maximize the diversity of trials occurrences for each unit. Each
    trial might occur multiple times or not at all depending on the number of available trials for
    the unit (``n``) compared to the required number pseudo-trials to generate (``n_pseudo``).
    Several cases are distinguished:

    - `n >= n_pseudo``: Trials are randomly selected without replacement. It includes the case
      ``n == n_pseudo``, where all trials are selected once.
    - ``n < n_pseudo``: Each trial is selected at least ``n_pseudo // n`` times. To reach the
      required number of trials, ``n_pseudo % n`` remaining trials are randomly selected without
      replacement.

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
    :func:`np.random.choice` :func:`np.repeat`
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
    Combine the trials which were retained across units to form pseudo-trials.

    Implementation
    --------------
    To mitigate redundancies in trials' pairings, combinations are built by randomly shuffling the
    trials for each unit. This approach is more straightforward than specifying an algorithm to
    maximize the diversity of mutual trials combinations across units.

    Parameters
    ----------
    trials : npt.NDArray
        Trials selected for each unit. Shape: ``(n_units, n_pseudo)``.

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


def hierarchical_bootstrap(
    counts: npt.NDArray[np.int_],
    n_pseudo_min=N_PSEUDO_MIN,
    alpha=ALPHA_BOOTSTRAP,
) -> npt.NDArray:
    """
    Execute the complete pipeline to generate pseudo-trials through hierarchical bootstrap.

    Parameters
    ----------
    counts : npt.NDArray[np.int_]
        Counts of available trials for each unit.
        Shape: ``(n_units,)``.

    Returns
    -------
    pseudo : npt.NDArray
        Pseudo-trials formed by combining trials across units.
        Shape: ``(n_units, n_pseudo)``.
    """
    n_pseudo = determine_pseudo_trial_number(counts, n_pseudo_min=n_pseudo_min, alpha=alpha)
    trials = np.array([pick_trials(n, n_pseudo) for n in counts])  # rows: units, columns: trials
    pseudo = combine_trials(trials)
    return pseudo
