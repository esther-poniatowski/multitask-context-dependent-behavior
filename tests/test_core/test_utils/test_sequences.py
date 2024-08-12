#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_utils.test_sequences` [module]

See Also
--------
:mod:`core.utils.sequences`: Tested module.
"""

import numpy as np
import pytest

from core.utils.sequences import unique, reverse_dict_container


seq = [1, 2, 2, 3, 3, 3, 4]
seq_unique = [1, 2, 3, 4]
input_seqs = [seq, tuple(seq), np.array(seq)]
expected_seqs = [list(seq_unique), tuple(seq_unique), np.array(seq_unique)]


@pytest.mark.parametrize(
    "sequence, expected_output",
    argvalues=[(seq, exp) for seq, exp in zip(input_seqs, expected_seqs)],
    ids=["list", "tuple", "ndarray"],
)
def test_unique(sequence, expected_output):
    """
    Test for :func:`core.utils.sequences.unique`.

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


def test_reverse_dict_container():
    """
    Test for :func:`core.utils.sequences.reverse_dict_container`
    with a valid container type.

    Test Inputs
    -----------
    dct: Dict[str, List[int]]
        Keys: Two strings.
        Values: Lists of integers, with one unique and one repeated element.

    Expected Output
    ---------------
    expected_dict: Dict[int, Set[str]]
        Keys: Three integers.
        Values: Lists containing the two strings.
    """
    input_dct = {
        "a": [1, 2],
        "b": [2, 3],
    }
    expected_dct = {
        1: ["a"],
        2: ["a", "b"],
        3: ["b"],
    }
    rev_dct = reverse_dict_container(input_dct)
    assert rev_dct == expected_dct, "Wrong output"
