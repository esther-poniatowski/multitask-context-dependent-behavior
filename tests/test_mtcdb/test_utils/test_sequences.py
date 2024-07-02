#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_utils.test_sequences` [module]

See Also
--------
:mod:`mtcdb.utils.sequences`: Tested module.
"""

import numpy as np
import pytest

from mtcdb.utils.sequences import unique

seq = [1, 2, 2, 3, 3, 3, 4]
seq_unique = [1, 2, 3, 4]
input_seqs = [seq, tuple(seq), np.array(seq)]
expected_seqs = [list(seq_unique), tuple(seq_unique), np.array(seq_unique)]
@pytest.mark.parametrize("sequence, expected_output",
                         argvalues = [(seq, exp) for seq, exp in zip(input_seqs, expected_seqs)],
                         ids = ["list", "tuple", "ndarray"])
def test_unique(sequence, expected_output):
    """
    Test for :func:`mtcdb.utils.sequences.unique`.

    Test Inputs
    -----------
    sequence: Iterable [test parameter]
        Simple sequence with duplicates, of one of the following types:
        ``list``, ``tuple``, ``ndarray``.
    
    Expected Outputs
    ----------------
    expected_output: Iterable
        Sequence with unique elements, of the same type as :obj:`sequence`.

    Implementation
    --------------
    For arrays, assess equality with :func:`np.array_equal` instead of ``==``.
	"""
    result = unique(sequence)
    assert isinstance(result, type(expected_output)), "Wrong type"
    if isinstance(sequence, np.ndarray):
        assert np.array_equal(result, expected_output), "Wrong output"
    else:
        assert result == expected_output, "Wrong output"
