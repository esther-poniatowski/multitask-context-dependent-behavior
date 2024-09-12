#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_preprocess.test_assign_folds` [module]

See Also
--------
:mod:`core.pipelines.preprocess.assign_folds`: Tested module.
"""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from core.pipelines.preprocess.assign_folds import FoldsAssigner


def test_assign_folds_no_stratification():
    """
    Test :meth:`FoldsAssigner.assign` without stratification.

    Test Inputs
    -----------
    n_samples = 6
    k = 3

    Expected Outputs
    ----------------
    Random fold assignment where each fold is assigned two samples.
    """
    n_samples = 6
    k = 3
    assigner = FoldsAssigner(k, n_samples)
    folds = assigner.assign()
    assert len(folds) == n_samples, f"Expected {n_samples} samples, Got {len(folds)}"
    assert np.unique(folds).size == k, f"Expected {k} folds, Got {np.unique(folds).size}"


def test_assign_folds_stratified_divisible():
    """
    Test :meth:`FoldsAssigner.assign` with stratification where the number of samples in each
    stratum is divisible by the number of folds.

    Test Inputs
    -----------
    n_samples = 9
    strata = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    k = 3

    Expected Outputs
    ----------------
    Each stratum should be perfectly split across all k folds.
    """
    n_samples = 9
    strata = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    k = 3
    assigner = FoldsAssigner(k, n_samples, strata)
    folds = assigner.assign()
    assert len(folds) == n_samples, f"Expected {n_samples} samples, Got {len(folds)}"
    for stratum in np.unique(strata):
        assert np.unique(folds[strata == stratum]).size == k, f"Stratum {stratum} not evenly split."


def test_assign_folds_stratified_non_divisible():
    """
    Test :meth:`FoldsAssigner.assign` with stratification where the number of samples in each
    stratum is not divisible by the number of folds.

    Test Inputs
    -----------
    n_samples = 6
    strata = [0, 0, 1, 1, 2, 2]
    k = 3

    Expected Outputs
    ----------------
    Strata should be split as evenly as possible across the folds.

    See Also
    --------
    :func:`numpy.bincount`: Used to count the number of samples in each fold for each stratum.
    """
    n_samples = 6
    strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    k = 3
    assigner = FoldsAssigner(k, n_samples, strata)
    folds = assigner.assign()
    assert len(folds) == n_samples, f"Expected {n_samples} samples, Got {len(folds)}"
    for stratum in np.unique(strata):
        fold_counts = np.bincount(folds[strata == stratum], minlength=k)
        assert fold_counts.max() - fold_counts.min() <= 1, f"Stratum {stratum} not evenly split."


def test_invalid_strata_shape():
    """
    Test :meth:`FoldsAssigner._validate_strata` raises ValueError for invalid strata shape.

    Test Inputs
    -----------
    n_samples = 6
    strata = np.array([0, 1])

    Expected Outputs
    ----------------
    ValueError is raised due to mismatch in strata shape.
    """
    n_samples = 6
    strata = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError):
        FoldsAssigner(k=3, n_samples=n_samples, strata=strata)


def test_folds_property_cache():
    """
    Test :attr:`FoldsAssigner.folds` caching mechanism.

    Test Inputs
    -----------
    n_samples = 6
    k = 3

    Expected Outputs
    ----------------
    The folds should be cached after the first access.
    Subsequent accesses should not recompute the fold assignments.
    """
    n_samples = 6
    k = 3
    assigner = FoldsAssigner(k, n_samples)
    folds_first = assigner.folds
    folds_second = assigner.folds
    assert_array_equal(folds_first, folds_second), "Folds are not cached properly."


def test_invalid_n_samples():
    """
    Test :meth:`FoldsAssigner.assign` raises ValueError if n_samples < k.

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
    with pytest.raises(ValueError):
        assigner = FoldsAssigner(k, n_samples)


def test_dynamic_n_samples_update():
    """
    Test dynamic update of `n_samples` and recomputation of folds.

    Test Inputs
    -----------
    Initial n_samples = 6
    Updated n_samples = 4

    Expected Outputs
    ----------------
    The folds should be recomputed when `n_samples` is updated.
    """
    # Initial fold assignment
    n_samples = 6
    k = 3
    assigner = FoldsAssigner(k, n_samples)
    initial_folds = assigner.folds
    assert len(initial_folds) == n_samples
    # Update n_samples and ensure folds are recomputed
    assigner.n_samples = 4
    updated_folds = assigner.folds
    assert len(updated_folds) == 4, "Folds were not updated after changing n_samples"


def test_seed_consistency():
    """
    Test fold assignment consistency when a seed is provided.

    Test Inputs
    -----------
    n_samples = 6
    k = 3
    seed = 42

    Expected Outputs
    ----------------
    Fold assignments should be consistent across runs with the same seed.
    """
    n_samples = 6
    k = 3
    seed = 42
    assigner_1 = FoldsAssigner(k, n_samples, seed=seed)
    assigner_2 = FoldsAssigner(k, n_samples, seed=seed)
    folds_1 = assigner_1.folds
    folds_2 = assigner_2.folds
    assert_array_equal(folds_1, folds_2), "Fold assignments are not consistent with the same seed"
