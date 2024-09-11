#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_preprocess.test_stratifier` [module]

See Also
--------
:mod:`core.pipelines.preprocess.stratify`: Tested module.
"""

import numpy as np
from numpy.testing import assert_array_equal as assert_array_eq
import pytest

from core.pipelines.preprocess.stratify import Stratifier


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
    stratifier = Stratifier(features)
    strata = stratifier.strata
    assert_array_eq(strata, expected_strata), f"Expected {expected_strata}, Got {strata}"


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
    with pytest.raises(ValueError):
        Stratifier(features)


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
    with pytest.raises(ValueError):
        Stratifier(features)
