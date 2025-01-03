#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_preprocess.test_hierarchical_bootstrap` [module]

See Also
--------
:mod:`core.preprocess.hierarchical_bootstrap`: Tested module.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_array_eq
import pytest

from core.preprocess.hierarchical_bootstrap import (
    pick_trials,
    hierarchical_bootstrap,
)


def test_pick_trials_equal():
    """
    Tests for :func:`pick_trials`, case where ``n == n_pseudo``.

    Test Inputs
    -----------
    n = 3
    n_pseudo = 3

    Expected Outputs
    ----------------
    Each trial number is selected exactly once.
    expected_count = np.array([1, 1, 1])

    See Also
    --------
    :func:`numpy.bincount`:
        Count the number of occurrences of each value in an array of non-negative integers.
        Output: Array where the index represents the unique values in the input array, and the
        corresponding value represents the count of occurrences.
        Example: ``np.bincount([0, 1, 1, 3])`` returns ``array([1, 2, 0, 1])``,
        which means that 0 occurs once, 1 occurs twice, 2 occurs zero times, and 3 occurs once.
    """
    n, n_pseudo = 3, 3
    trials = pick_trials(n, n_pseudo)
    count_occurrences = np.bincount(trials, minlength=n)
    expected_count = np.array([1, 1, 1])
    assert_array_eq(count_occurrences, expected_count), f"Expected equal counts"


def test_pick_trials_greater():
    """
    Tests for :func:`pick_trials`, case where ``n > n_pseudo``.

    Test Inputs
    -----------
    n = 5
    n_pseudo = 3

    Expected Outputs
    ----------------
    n_pseudo trials are selected only once.
    n - n_pseudo trials are not selected.
    expected_nb_1 = n_pseudo
    expected_nb_0 = n - n_pseudo
    """
    n, n_pseudo = 5, 3
    trials = pick_trials(n, n_pseudo)
    # Count the number of occurrences of each trial index
    count_occurrences = np.bincount(trials, minlength=n)
    # Count the number of 1 and O in `counts` itself
    nb_1 = np.count_nonzero(count_occurrences == 1)
    nb_0 = np.count_nonzero(count_occurrences == 0)
    expected_1 = n_pseudo
    expected_0 = n - n_pseudo
    assert nb_1 == expected_1, f"Expected {expected_1}, Got {nb_1}"
    assert nb_0 == expected_0, f"Expected {expected_0}, Got {nb_0}"


def test_pick_trials_smaller():
    """
    Tests for :func:`pick_trials`, case where ``n < n_pseudo``.

    Test Inputs
    -----------
    n = 4
    n_pseudo = 11

    Expected Outputs
    ----------------
    q = n_pseudo // n = 11 // 4 = 2
    r = n_pseudo % n = 11 % 4 = 3
    Each trial is selected at least q times and r trials are selected one more time.
    Thus n - r trials are selected q times and r trials are selected q + 1 times.
    expected_nb_q = n - r
    expected_nb_q_plus_1 = r
    """
    n, n_pseudo = 3, 8
    q, r = divmod(n_pseudo, n)
    trials = pick_trials(n, n_pseudo)
    # Count the number of occurrences of each trial index
    counts = np.bincount(trials, minlength=n)
    # Count the number of trials selected q times and q + 1 times
    nb_q = np.count_nonzero(counts == q)
    nb_q_plus_1 = np.count_nonzero(counts == q + 1)
    expected_q = n - r
    expected_q_plus_1 = r
    assert nb_q == expected_q, f"Expected {expected_q}, Got {nb_q}"
    assert nb_q_plus_1 == expected_q_plus_1, f"Expected {expected_q_plus_1}, Got {nb_q_plus_1}"


def test_hierarchical_bootstrap():
    """
    Tests for :func:`hierarchical_bootstrap`.

    Test Inputs
    -----------
    counts = np.array([3, 4, 5, 6, 7])
    alpha = 0.5
    n_pseudo_min = 4

    Expected Outputs
    ----------------
    Number of trials to generate:
    n_pseudo = mean(min+max) = (3+7)/2 = 5 > n_pseudo_min
    n_units = 5
    """
    n_min, n_max = 3, 7
    n_pseudo_min = 4
    alpha = 0.5
    expected_n = 5
    counts = np.arange(n_min, n_max + 1)  # uniform from `n_min` to `n_max`
    n_units = len(counts)
    expected_shape = (n_units, expected_n)
    pseudo = hierarchical_bootstrap(counts, n_pseudo_min, alpha)
    assert pseudo.shape[0] == n_units, f"Expected {n_units} units, Got {pseudo.shape[0]}"
    assert pseudo.shape[1] == expected_n, f"Expected {expected_n} trials, Got {pseudo.shape[1]}"
    assert pseudo.shape == expected_shape, f"Expected shape {expected_shape}, Got {pseudo.shape}"
    assert np.all(pseudo < counts[:, None]), f"Items out of bounds"
