#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.exclude` [module]

Classes
-------
`Stratifier`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# pylint: disable=useless-parent-delegation

from typing import List, Any

import numpy as np

from core.processors.base_processor import Processor


class Excluder(Processor):
    """
    Exclude a set of elements from another set.

    Processing Arguments
    --------------------
    candidates : List[Any]
        Candidate elements from which some members might be excluded.
    excluded : List[Any]
        Elements to exclude from the candidate set.

    Returns
    -------
    retained : List[Any]
        Elements retained after exclusion.

    Examples
    --------
    Exclude a set of elements from another set:

    >>> excluder = Excluder()
    >>> candidates = [1, 2, 3, 4, 5]
    >>> excluded = [2, 4]
    >>> retained = excluder.process(candidates, excluded)
    >>> retained
    [1, 3, 5]

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random: bool = False

    # --- Processing Methods -----------------------------------------------------------------------

    def _process(self, candidates=None, excluded=None, **input_data) -> List[Any]:
        """Implement the template method called in the base class `process` method."""
        assert candidates is not None and excluded is not None, "Missing required arguments."
        retained = [element for element in candidates if element not in excluded]
        return retained
