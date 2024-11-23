#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`utils.misc.sequences` [module]

Utilities for sequences or iterables (lists, dictionaries, tuples, array-like...).

Functions
---------
`unique`
`reverse_dict_container`
`exclude_by_difference`
`filter_by_predicate`s

See Also
--------
`tests_core.test_utils.test_sequences`: Unit tests for this module.
"""
from collections import defaultdict
from collections.abc import Hashable  # pylint: disable=unused-import
from typing import List, Tuple, Dict, Mapping, Iterable, TypeVar, Callable, Any


import numpy as np


I = TypeVar("I", np.ndarray, List, Tuple)
"""Type variable for input sequences."""

K = TypeVar("K")
"""Type variable for keys in dictionaries."""

V = TypeVar("V")
"""Type variable for values in dictionaries."""


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
    :func:`numpy.unique`: Keep unique elements in numpy arrays, in *ascending* order.

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
            raise TypeError("Unsupported sequence type")


def reverse_dict_container(dct: Mapping[K, Iterable[V]]) -> Dict[V, List[K]]:
    """
    Reverse a dictionary with container values.

    Parameters
    ----------
    dct: Dict
        Dictionary with container values.
        Keys: Any hashable type.
        Values: Iterable of hashable types.

    Returns
    -------
    Mapping
        Reverse dictionary.
        Keys: Unique values from the container values in the input dictionary.
        Values: List storing the initial keys from the input dictionary.

    Example
    -------
    >>> reverse_dict_container({'a': [1, 2], 'b': [2, 3]})
    {1: ['a'], 2: ['a', 'b'], 3: ['b']}

    See Also
    --------
    :func:`collections.defaultdict`: Dictionary with default values for missing keys.

        Parameter: ``default_factory``, function that automatically creates a default value whenever
        a key that does not exist in the dictionary is accessed.

        Here, is used to initialize each new key in the reversed dictionary with an empty container
        as its value. It avoids the need to check if the key already exists.
    """
    rev_dct: Dict[V, List[K]] = defaultdict(list)  # list to append values
    for key, values in dct.items():
        for value in values:
            rev_dct[value].append(key)
    return rev_dct


def exclude_by_difference(candidates: Iterable, intruders: Iterable) -> List[Any]:
    """
    Exclude a set of elements from another set.

    Arguments
    ---------
    candidates : Iterable
        Candidate elements from which some members might be excluded.
    intruders : Iterable
        Elements to exclude from the candidate set.

    Returns
    -------
    retained : List[Any]
        Elements retained after exclusion.

    Examples
    --------
    Exclude a set of elements from another set:

    >>> candidates = [1, 2, 3, 4, 5]
    >>> intruders = [2, 4]
    >>> retained = Excluder.exclude_by_difference(candidates, intruders)
    >>> retained
    [1, 3, 5]
    """
    return [element for element in candidates if element not in intruders]


def filter_by_predicate(values: Iterable, predicate: Callable) -> List[int]:
    """
    Retain indices of elements that satisfy a given predicate.

    Arguments
    ---------
    values : Iterable
        Values provided to match the predicate.
    predicate : Callable
        Function that takes a single argument and returns a boolean, indicating whether an
        element satisfies the condition.

    Returns
    -------
    idx : List[int]
        Indices of the elements that satisfy the predicate.
    """
    return [idx for idx, value in enumerate(values) if predicate(value)]
