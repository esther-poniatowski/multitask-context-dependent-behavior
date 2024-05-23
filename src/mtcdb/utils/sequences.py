#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.utils.sequences` [module]
=====================================

Utility functions for sequences.
"""

import numpy as np
from typing import Iterable, TypeVar

I = TypeVar('I', bound='Iterable')


def unique(sequence: I) -> I:
    """
    Filter out repeated elements in a sequence, ordered by their *first occurrence*.
    
    Parameters
    ----------
    sequence: Iterable
        Sequence which may contain repeated elements.
    
    Returns
    -------
    Iterable
        Filtered sequence whose elements occur only once.
    
    Raises
    ------
    ValueError
        If the sequence type is not supported.
    
    See Also
    --------
    numpy.unique: Keep unique elements in numpy arrays, in *ascending* order.

    Implementation
    --------------
    For numpy arrays, :func:`numpy.unique()` with ``return_index=True``
    provides the indices of the first occurrences of the unique elements.
    Indexing by ``np.sort(idx)`` ensures to extract the elements in the order of their appearance.

    For other sequences, :meth:`dict.fromkeys()` exploits the property
    that dictionaries maintain insertion order but do not allow duplicate keys.
    """
    if isinstance(sequence, np.ndarray):
        _, idx = np.unique(sequence, return_index=True)
        return sequence[np.sort(idx)]
    else:
        unique_items = dict.fromkeys(sequence).keys()
        if isinstance(sequence, list):
            return list(unique_items)
        elif isinstance(sequence, tuple):
            return tuple(unique_items)
        else:
            raise ValueError("Unsupported sequence type")
