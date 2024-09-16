#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_preprocess.test_assign_folds` [module]

See Also
--------
:mod:`core.processors.preprocess.assign_folds`: Tested module.
"""
# Disable error code for access to protected members:
# pylint: disable=protected-access

# Disable error code for expression not being assigned:
# pylint: disable=expression-not-assigned

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from core.processors.preprocess.assign_folds import FoldAssigner


def test_missing_inputs():
    """
    Test :meth:`FoldAssigner._validate` when both `strata` and `n_samples` are missing.

    Test Inputs
    -----------
    n_samples = None
    strata = None

    Expected Outputs
    ----------------
    ValueError raised due to missing inputs.
    """
    k = 3
    assigner = FoldAssigner(k)
    # Via `_validate` method (subclass)
    with pytest.raises(ValueError):
        assigner._validate(n_samples=None, strata=None)
    # Via `process` method (base class)
    with pytest.raises(ValueError):
        assigner.process()


def test_both_inputs_provided():
    """
    Test :meth:`FoldAssigner._validate` when both `strata` and `n_samples` are provided.

    Test Inputs
    -----------
    n_samples = 6
    strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    Expected Outputs
    ----------------
    ValueError raised due to extra inputs.
    """
    k = 3
    n_samples = 6
    strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assigner = FoldAssigner(k)
    with pytest.raises(ValueError):
        assigner._validate(n_samples=n_samples, strata=strata)


def test_invalid_n_samples():
    """
    Test :meth:`FoldAssigner._validate_n_samples` raises ValueError if n_samples < k.

    Test Inputs
    -----------
    n_samples = 4
    k = 5

    Expected Outputs
    ----------------
    ValueError is raised as n_samples cannot be less than k.
    """
    n_samples = 4
    k = 5
    assigner = FoldAssigner(k)
    with pytest.raises(ValueError):
        assigner._validate_n_samples(n_samples)


def test_invalid_strata_shape():
    """
    Test :meth:`FoldAssigner._validate_strata` raises ValueError for invalid strata shape.

    Test Inputs
    -----------
    k = 3
    strata = np.array([0, 1]) (length < k)

    Expected Outputs
    ----------------
    ValueError is raised due to mismatch in strata shape.
    """
    k = 3
    strata = np.array([0, 1], dtype=np.int64)
    assigner = FoldAssigner(k)
    with pytest.raises(ValueError):
        assigner._validate_strata(strata=strata)


def test_default_strata():
    """
    Test default strata assignment when `strata` is None, with both
    :meth:`FoldAssigner._default` and :meth:`FoldAssigner.process`.

    Test Inputs
    -----------
    n_samples = 6
    strata = None or not provided

    Expected Outputs
    ----------------
    Default strata assignment where all samples are treated as belonging to a single stratum.
    """
    k = 3
    n_samples = 6
    expected_strata = np.zeros(n_samples, dtype=np.int64)
    # Via `_default` method (subclass)
    assigner = FoldAssigner(k)
    input_data = assigner._default(n_samples=n_samples, strata=None)
    strata = input_data["strata"]
    assert strata.size == n_samples, f"strata.size = {strata.size} != {n_samples}"
    assert_array_equal(strata, expected_strata)
    # Via `process` method (base class)
    assigner = FoldAssigner(k)
    assigner.process(n_samples=n_samples)
    strata = assigner.strata
    assert strata.size == n_samples, f"strata.size = {strata.size} != {n_samples}"
    assert_array_equal(strata, expected_strata)


def test_default_n_samples():
    """
    Test default n_samples assignment when `n_samples` is None, with both
    :meth:`FoldAssigner._default` and :meth:`FoldAssigner.process`.

    Test Inputs
    -----------
    n_samples = None or not provided
    strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    Expected Outputs
    ----------------
    Default n_samples assignment where the number of samples is equal to the length of strata.
    """
    k = 3
    strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    # Via `_default` method (subclass)
    assigner = FoldAssigner(k)
    input_data = assigner._default(n_samples=None, strata=strata)
    n_samples = input_data["n_samples"]
    assert n_samples == strata.size, f"n_samples = {n_samples} != {strata.size}"
    # Via `process` method (base class)
    assigner = FoldAssigner(k)
    assigner.process(strata=strata)
    n_samples = assigner.n_samples
    assert n_samples == strata.size, f"n_samples = {n_samples} != {strata.size}"


def test_folds_property_cache():
    """
    Test :attr:`FoldAssigner.folds` caching mechanism.

    Test Inputs
    -----------
    n_samples = 6

    Expected Outputs
    ----------------
    After computation, folds should be stored in the internal attribute `_folds` and accessible via
    the public property `folds`.
    Check whether the content of those attributes is not an empty array (default value).

    Warning
    -------
    This test only ensures that the cache is working as expected. It does not check the correctness
    of the values computed for the fold assignment.
    """
    k = 3
    n_samples = 6
    assigner = FoldAssigner(k)
    output = assigner.process(n_samples=n_samples)
    folds_returned = output["folds"]
    folds_internal = assigner._folds
    folds_public = assigner.folds
    assert len(folds_returned) == n_samples, f"len(folds) = {len(folds_returned)} != {n_samples}"
    assert len(folds_internal) == n_samples, f"len(folds) = {len(folds_internal)} != {n_samples}"
    assert len(folds_public) == n_samples, f"len(folds) = {len(folds_public)} != {n_samples}"
    assert_array_equal(folds_returned, folds_internal)
    assert_array_equal(folds_returned, folds_public)


def test_assign_folds_no_stratification():
    """
    Test :meth:`FoldAssigner.assign` without stratification.

    Test Inputs
    -----------
    k = 3
    n_samples = 6

    Expected Outputs
    ----------------
    Random fold assignment where each fold is assigned two samples.
    """
    k = 3
    n_samples = 6
    assigner = FoldAssigner(k)
    assigner.process(n_samples=n_samples)
    folds = assigner.folds
    assert len(folds) == n_samples, f"Expected {n_samples} samples, Got {len(folds)}"
    assert np.unique(folds).size == k, f"Expected {k} folds, Got {np.unique(folds).size}"


def test_assign_folds_stratified_divisible():
    """
    Test :meth:`FoldAssigner.assign` with stratification where the number of samples in each
    stratum is divisible by the number of folds.

    Test Inputs
    -----------
    k = 3
    strata = [0, 0, 0, 1, 1, 1, 2, 2, 2] (3 samples per stratum = 9 samples)

    Expected Outputs
    ----------------
    Each stratum should be perfectly split across all k folds.
    """
    k = 3
    strata = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    n_samples = strata.size
    assigner = FoldAssigner(k)
    assigner.process(strata=strata)
    folds = assigner.folds
    assert len(folds) == n_samples, f"Expected {n_samples} samples, Got {len(folds)}"
    for stratum in np.unique(strata):
        assert np.unique(folds[strata == stratum]).size == k, f"Stratum {stratum} not evenly split."


def test_assign_folds_stratified_non_divisible():
    """
    Test :meth:`FoldAssigner.assign` with stratification where the number of samples in each
    stratum is not divisible by the number of folds.

    Test Inputs
    -----------
    k = 3
    n_samples = 6
    strata = [0, 0, 1, 1, 2, 2] (2 samples per stratum = 6 samples)

    Expected Outputs
    ----------------
    Strata should be split as evenly as possible across the folds.

    See Also
    --------
    :func:`numpy.bincount`: Used to count the number of samples in each fold for each stratum.
    """
    k = 3
    strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    n_samples = strata.size
    assigner = FoldAssigner(k)
    assigner.process(strata=strata)
    folds = assigner.folds
    assert len(folds) == n_samples, f"Expected {n_samples} samples, Got {len(folds)}"
    for stratum in np.unique(strata):
        fold_counts = np.bincount(folds[strata == stratum], minlength=k)
        assert fold_counts.max() - fold_counts.min() <= 1, f"Stratum {stratum} not evenly split."


def test_seed_consistency():
    """
    Test fold assignment consistency when a seed is provided.

    Test Inputs
    -----------
    k = 3
    n_samples = 6
    seed = 42

    Expected Outputs
    ----------------
    Fold assignments should be consistent across runs with the same seed.
    """
    n_samples = 6
    k = 3
    seed = 42
    assigner = FoldAssigner(k)
    assigner.process(n_samples=n_samples, seed=seed)
    folds_1 = assigner.folds
    assigner.process(n_samples=n_samples, seed=seed)  # same seed
    folds_2 = assigner.folds
    assert_array_equal(folds_1, folds_2), "Fold assignments are not consistent with the same seed"
