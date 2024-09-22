#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_preprocess.test_assign_ensembles` [module]

See Also
--------
:mod:`core.processors.preprocess.assign_ensembles`: Tested module.
"""
# Disable error code for access to protected members:
# pylint: disable=protected-access

# Disable error code for expression not being assigned:
# pylint: disable=expression-not-assigned

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from core.processors.preprocess.assign_ensembles import EnsembleAssigner


def test_invalid_n_units():
    """
    Test :meth:`EnsembleAssigner._validate_n_units`.

    Test Inputs
    -----------
    n_units = 2
    ensemble_size = 3

    Expected Outputs
    ----------------
    ValueError raised as `n_units` cannot be less than ensemble_size.
    """
    n_units = 2
    ensemble_size = 3
    assigner = EnsembleAssigner(ensemble_size)
    with pytest.raises(ValueError):
        assigner._validate(n_units=n_units)


@pytest.mark.parametrize(
    "n_units, ensemble_size, n_ensembles_max, n_expected",
    argvalues=[(10, 5, None, 2), (10, 6, None, 2), (10, 6, 1, 1)],
    ids=["divisible", "non-divisible", "max-ensembles"],
)
def test_determine_n_ensembles(n_units, ensemble_size, n_ensembles_max, n_expected):
    """
    Parametrized test for computing the number of ensembles based on the number of units.

    Test Inputs
    -----------
    n_units: int
        Number of units.
    ensemble_size: int
        Size of each ensemble.
    n_ensembles_max: int or None
        Maximum number of ensembles (optional).

    Expected Outputs
    ----------------
    n_expected: int
        Expected number of ensembles.
    """
    assigner = EnsembleAssigner(ensemble_size=ensemble_size, n_ensembles_max=n_ensembles_max)
    assigner.n_units = n_units  # assign manually
    assigner.eval_n_ensembles()
    n_ensembles = assigner.n_ensembles
    assert n_ensembles == n_expected, f"Expected {n_expected} ensembles, Got {n_ensembles}"


def test_single_ensemble():
    """
    Test ensemble assignment when ``n_units == ensemble_size``.

    Test Inputs
    -----------
    n_units = 4
    ensemble_size = 4

    Expected Outputs
    ----------------
    Only one ensemble is created.
    """
    n_units = 4
    ensemble_size = 4
    assigner = EnsembleAssigner(ensemble_size)
    assigner.n_units = n_units  # assign manually
    assigner.assign()
    ensembles = assigner.ensembles
    assert ensembles.shape[0] == 1, f"Expected 1 ensemble, Got {ensembles.shape[0]}"
    assert ensembles.size == n_units, f"Expected {n_units} units, Got {ensembles.size}"


def test_multiple_ensembles():
    """
    Test ensemble assignment when ``n_units > ensemble_size`` and ``n_units % ensemble_size != 0``.

    Test Inputs
    -----------
    n_units = 10
    ensemble_size = 5

    Expected Outputs
    ----------------
    Creation of 2 ensembles with disjoint sets of units.
    """
    n_units = 10
    ensemble_size = 5
    n_ens_expected = 2
    assigner = EnsembleAssigner(ensemble_size)
    assigner.n_units = n_units  # assign manually
    assigner.assign()
    ensembles = assigner.ensembles
    # Test dimensions
    shape = ensembles.shape
    assert shape[0] == n_ens_expected, f"Expected {n_ens_expected} ensembles, Got {shape[0]}"
    assert shape[1] == ensemble_size, f"Expected ensemble size {ensemble_size}, Got {shape[1]}"
    # Test that each ensemble has unique units
    for ensemble in ensembles:
        assert len(ensemble) == len(np.unique(ensemble)), "Ensemble contains duplicate units"
    # Test that ensembles are disjoint
    flattened_ensembles = np.concatenate(ensembles)
    assert len(flattened_ensembles) == len(
        np.unique(flattened_ensembles)
    ), "Ensembles are not disjoint"


def test_last_ensemble_filling():
    """
    Test the behavior when the last ensemble is filled from previous ones when ``n_units %
    ensemble_size != 0``.

    Test Inputs
    -----------
    n_units = 10
    ensemble_size = 4

    Expected Outputs
    ----------------
    Creation of 3 ensembles with 4 units each.
    The two first ensembles have disjoint sets of units.
    The last ensemble has 2 units which are not in the first two ensembles and 2 units from the
    first two ensembles.
    """
    n_units = 10
    ensemble_size = 4
    n_ens_expected = 3
    assigner = EnsembleAssigner(ensemble_size)
    assigner.n_units = n_units  # assign manually
    assigner.assign()
    ensembles = assigner.ensembles
    # Test dimensions
    shape = ensembles.shape
    assert shape[0] == n_ens_expected, f"Expected {n_ens_expected} ensembles, Got {shape[0]}"
    assert shape[1] == ensemble_size, f"Expected ensemble size {ensemble_size}, Got {shape[1]}"
    # Test that each ensemble has unique units
    for ensemble in ensembles:
        assert len(ensemble) == len(np.unique(ensemble)), "Ensemble contains duplicate units"
    # Test that the first two ensembles are disjoint
    for i, ensemble in enumerate(ensembles[:-1]):
        for other_ensemble in ensembles[i + 1 : -1]:
            assert len(np.intersect1d(ensemble, other_ensemble)) == 0, "Ensembles are not disjoint"
    # Test that the last ensemble shares 2 units from the first two ensembles
    ens0, ens1, ens2 = ensembles
    shared_units_0 = np.intersect1d(ens0, ens2)
    shared_units_1 = np.intersect1d(ens1, ens2)
    n_shared = len(shared_units_0) + len(shared_units_1)
    assert n_shared == 2, f"Expected 2 shared units, Got {n_shared}"


def test_exact_split():
    """
    Test ensemble assignment when ``n_units % ensemble_size == 0``.

    Test Inputs
    -----------
    n_units = 12
    ensemble_size = 4

    Expected Outputs
    ----------------
    Creation of 3 ensembles with 4 units each.
    """
    n_units = 12
    ensemble_size = 4
    n_ens_expected = 3
    assigner = EnsembleAssigner(ensemble_size)
    assigner.n_units = n_units  # assign manually
    assigner.assign()
    ensembles = assigner.ensembles
    # Test dimensions
    shape = ensembles.shape
    assert shape[0] == n_ens_expected, f"Expected {n_ens_expected} ensembles, Got {shape[0]}"
    assert shape[1] == ensemble_size, f"Expected ensemble size {ensemble_size}, Got {shape[1]}"
    # Test that each ensemble has unique units
    for ensemble in ensembles:
        assert len(ensemble) == len(np.unique(ensemble)), "Ensemble contains duplicate units"
    # Test that ensembles are disjoint
    flattened_ensembles = np.concatenate(ensembles)
    assert len(flattened_ensembles) == len(
        np.unique(flattened_ensembles)
    ), "Ensembles are not disjoint"


def test_seed_consistency():
    """
    Test ensemble assignment consistency when a seed is provided.

    Test Inputs
    -----------
    n_units = 10
    ensemble_size = 4
    seed = 42

    Expected Outputs
    ----------------
    Ensemble assignments should be consistent across runs with the same seed.
    """
    n_units = 10
    ensemble_size = 4
    seed = 42
    assigner = EnsembleAssigner(ensemble_size)
    assigner.process(n_units=n_units, seed=seed)
    ensembles_1 = assigner.ensembles
    assigner.process(n_units=n_units, seed=seed)  # same seed
    ensembles_2 = assigner.ensembles
    assert_array_equal(
        ensembles_1, ensembles_2
    ), "Ensemble assignments are not consistent with the same seed"
