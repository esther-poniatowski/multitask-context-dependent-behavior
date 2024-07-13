#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_preprocess.test_hierarchical_bootstrap` [module]

See Also
--------
:mod:`mtcdb.preprocess.hierarchical_bootstrap`: Tested module.
"""

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_almost_equal as assert_array_eq
import pytest

from mtcdb.preprocess.hierarchical_bootstrap import determine_pseudo_trial_number, pick_trials


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
    expected_count = np.array([1, 1, 1])
    actual = pick_trials(n, n_pseudo)
    assert_array_eq(np.bincount(actual), expected_count), f"Expected equal counts"


def test_pick_trials_greater():
    """
    Tests for :func:`pick_trials`, case where ``n > n_pseudo``.

    Test Inputs
    -----------
    n = 5
    n_pseudo = 3

    Expected Outputs
    ----------------
    n_pseudo trials are selected at only once.
    n - n_pseudo trials are not selected.
    expected_nb_1 = n_pseudo
    expected_nb_0 = n - n_pseudo
    """
    n, n_pseudo = 5, 3
    expected_1 = n_pseudo
    expected_0 = n - n_pseudo
    actual = pick_trials(n, n_pseudo)
    counts = np.bincount(actual)
    # Count the number of 1 and O in `counts``
    nb_1 = np.count_nonzero(counts == 1)
    nb_0 = np.count_nonzero(counts == 0)
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
    expected_q = n - r
    expected_q_plus_1 = r
    actual = pick_trials(n, n_pseudo)
    counts = np.bincount(actual)
    nb_q = np.count_nonzero(counts == q)
    nb_q_plus_1 = np.count_nonzero(counts == q + 1)
    assert nb_q == expected_q, f"Expected {expected_q}, Got {nb_q}"
    assert nb_q_plus_1 == expected_q_plus_1, f"Expected {expected_q_plus_1}, Got {nb_q_plus_1}"
