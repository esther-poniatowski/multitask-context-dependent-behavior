#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_utils.test_sequences` [module]
====================================================

Tests for the module :mod:`mtcdb.utils.sequences`.
"""

import numpy as np
import pytest

from mtcdb.utils.sequences import unique

sequence = [1, 2, 2, 3, 3, 3, 4] # sequence with duplicates
expected_output = [1, 2, 3, 4]
params = [(convert(sequence), convert(expected_output)) for convert in [list, tuple, np.array]] 
@pytest.mark.parametrize("sequence, expected_output", params, ids=["list", "tuple", "np.array"])
def test_unique(sequence, expected_output):
    """
    Test for :func:`mtcdb.utils.sequences.unique`.

    Test Inputs
    -----------
    sequence: Iterable [test parameter]
        One simple sequence is tested for each type:
        ``list``, ``tuple``, ``ndarray``.
    
    Implementation
    --------------
    For arrays, assess equality with :func:`np.array_equal` instead of ``==``.
	"""
    result = unique(sequence)
    assert type(result) == type(expected_output), "Wrong type"
    if isinstance(sequence, np.ndarray):
        assert np.array_equal(result, expected_output), "Wrong output"
    else:
        assert result == expected_output, "Wrong output"

