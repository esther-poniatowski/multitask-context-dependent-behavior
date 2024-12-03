#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_components.core_data` [module]

Classes
-------
CoreData
CoreIndices
CoreRates
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=abstract-method
# Scope: `CoreData` class
# Reason: The methods `flip` and `rollaxis` are not implemented in the base class `DataComponent`.
# --------------------------------------------------------------------------------------------------

from types import MappingProxyType

import numpy as np

from core.data_components.base_data_component import DataComponent
from core.data_components.core_metadata import MetaDataField


class CoreData(DataComponent):
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


class CoreIndices(CoreData):
    """
    Core component referencing bare indices (for events, trials...).

    Attributes
    ----------
    scope : str
        Frame of reference in which indices should be interpreted.

    Examples
    --------
    Common values for the `scope` attribute:

    - "experiment": Indices correspond to trials within an experimental session.
    - "population": Indices correspond to units within a neural population.
    - "dataset": Indices correspond to samples in a specific dataset.
    """

    DTYPE = np.dtype("int64")
    SENTINEL = -1
    METADATA = MappingProxyType({"scope": MetaDataField(str, "")})


class CoreTimes(CoreData):
    """
    Core component containing time values.

    Attributes
    ----------
    origin : str
        Time origin relative to which time stamps should be interpreted.
    unit : str, default="sec"
        Time unit in which time stamps are expressed.

    Examples
    --------
    Common values for the `reference` attribute:

    - "block": Time stamps are relative to the beginning of each block of trials.
    - "session": Time stamps are relative to the beginning of the session.
    - "trial": Time stamps are relative to the beginning of each trial.

    Common values for the `unit` attribute: "sec" (seconds), "msec" (milliseconds).
    """

    DTYPE = np.dtype("float64")
    SENTINEL = np.nan
    METADATA = MappingProxyType(
        {
            "origin": MetaDataField(str, ""),
            "unit": MetaDataField(str, "sec"),
        }
    )


class CoreRates(CoreData):
    """
    Core component containing firing rates.

    Attributes
    ----------
    unit : str, default="spikes/sec"
        Rate unit in which values are expressed.
    """

    DTYPE = np.dtype("float64")
    SENTINEL = np.nan
    METADATA = MappingProxyType({"unit": MetaDataField(str, "spikes/sec")})
