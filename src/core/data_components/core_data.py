#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_components.core_data` [module]

Classes
-------
CoreData
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=abstract-method
# Scope: `CoreData` class
# Reason: The methods `flip` and `rollaxis` are not implemented in the base class `DataComponent`.
# --------------------------------------------------------------------------------------------------

from core.data_components.core_dimensions import Dimensions
from core.data_components.base_data_component import DataComponent, Dtype


class CoreData(DataComponent[Dtype]):
    """
    Core component of a data structure, containing data values to analyze or companion labels.

    Each object behaves like a `numpy.ndarray`, with additional dimension annotations (names).

    Arguments
    ---------
    values : np.ndarray
        Actual values to analyze or store.
    **metadata : Any
        Additional attributes to store alongside the values. The names of the attributes should be
        specified in the class attribute `METADATA`.

    Notes
    -----
    So far, this class does not implement any additional behavior compared to the base class (unlike
    the `Coordinate` class which handles the interaction with the `Attribute` class). It is used to
    provide a more specific type hint for the core data component of a data structure.

    See Also
    --------
    `DataComponent` : Base class for all data components.
    """

    # Declare custom attributes at the class level to specify type hints
    dims: Dimensions
