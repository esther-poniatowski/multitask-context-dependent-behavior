#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_preprocess.test_stratifier` [module]

See Also
--------
:mod:`core.processors.preprocess.stratify`: Tested module.
"""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from core.processors.preprocess.stratify import Stratifier


def test_invalid_feature_dimensions():
    """
    Test :meth:`Stratifier._validate_features` raises ValueError for invalid dimensions.

    Test Inputs
    -----------
    features: Arrays with invalid dimensions (e.g., 2D array).

    Expected Outputs
    ----------------
    ValueError is raised due to invalid feature dimensions.
    """
    features = [np.array([[1, 2], [3, 4]], dtype=np.int64)]  # 2D array, invalid
    stratifier = Stratifier()
    with pytest.raises(ValueError):
        stratifier._validate_features(features)


def test_invalid_feature_types():
    """
    Test :meth:`Stratifier._validate_features` raises ValueError for invalid types.

    Test Inputs
    -----------
    features: Arrays with invalid types.

    Expected Outputs
    ----------------
    ValueError is raised due to invalid feature types.
    """
    features = [np.array([1, 2, 3], dtype=object)]  # Invalid type
    stratifier = Stratifier()
    with pytest.raises(ValueError):
        stratifier._validate_features(features)


def test_unequal_number_of_samples():
    """
    Test :meth:`Stratifier._validate_features` raises ValueError for unequal number of samples.

    Test Inputs
    -----------
    features: Arrays with unequal number of samples.

    Expected Outputs
    ----------------
    ValueError is raised due to unequal number of samples across features.
    """
    features = [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
    ]
    stratifier = Stratifier()
    with pytest.raises(ValueError):
        stratifier._validate_features(features)


def test_stratify():
    """
    Test :meth:`Stratifier.stratify`.

    Test Inputs
    -----------
    features: Integer and string arrays for 3 samples.

    Expected Outputs
    ----------------
    Two samples have identical feature combinations.
    Expected strata: [0, 0, 1]
    """
    features = [
        np.array([1, 1, 2], dtype=np.int64),
        np.array([0.1, 0.1, 0.2], dtype=np.float64),
        np.array(["A", "A", "B"], dtype=np.str_),
    ]
    expected_strata = np.array([0, 0, 1], dtype=np.int64)
    stratifier = Stratifier()
    stratifier.features = features  # set features manually
    stratifier.stratify()
    strata = stratifier.strata
    assert_array_equal(strata, expected_strata), f"Expected {expected_strata}, Got {strata}"


def test_process():
    """
    Test the full processing pipeline of :class:`Stratifier`.

    Test Inputs
    -----------
    Initial and updated feature arrays.

    Expected Outputs
    ----------------
    Strata are updated correctly when `features` are changed dynamically.
    """
    stratifier = Stratifier()
    # Create Stratifier and check initial strata
    features_1 = [np.array([1, 1, 1], dtype=np.int64)]
    expected_1 = np.array([0, 0, 0], dtype=np.int64)
    stratifier.process(features=features_1)
    strata_1 = stratifier.strata
    assert_array_equal(strata_1, expected_1), f"Expected {expected_1}, Got {strata_1}"
    # Update features
    features_2 = [np.array([1, 1, 2, 2], dtype=np.int64)]
    expected_2 = np.array([0, 0, 1, 1], dtype=np.int64)
    stratifier.process(features=features_2)
    strata_2 = stratifier.strata
    assert_array_equal(strata_2, expected_2), f"Expected {expected_2}, Got {strata_2}"
