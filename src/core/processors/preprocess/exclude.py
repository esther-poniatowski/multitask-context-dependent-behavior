#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.exclude` [module]

Classes
-------
`Excluder`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# pylint: disable=useless-parent-delegation

from typing import List, Any, Iterable

import numpy as np


Counts = Iterable[int]


class Excluder:
    """
    Utilities to exclude elements from a set from various criteria.

    Methods
    -------
    `exclude_by_difference`
    `exclude_from_counts`

    Notes
    -----
    This class is not a subclass of `Processor`. It only provides static utilities.
    """

    is_random: bool = False

    # --- Processing Methods -----------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def exclude_from_counts(counts: Counts, n_min: int) -> np.ndarray:
        """
        Exclude elements associated with counts inferior to a required minimum.

        Arguments
        ---------
        counts : Counts
            Counts provided by each element in the (implicit) set to consider.
        n_min : int
            Minimum count required to retain an element.

        Returns
        -------
        idx : np.ndarray
            Indices of the element to retain.

        Implementation
        --------------
        The function `np.where` returns a tuple of indices where the condition is met, where each
        element corresponds to a dimension in the input array. Here, the input array is 1D, so
        a single element is returned. To extract the indices, the first element of the tuple is
        used.
        """
        return np.where(np.array(counts) >= n_min)[0]
