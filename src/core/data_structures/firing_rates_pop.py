#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.firing_rates_pop` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Union

import numpy as np

from core.data_structures.core_data import DimName, CoreData
from core.coordinates.bio import CoordUnit
from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim
from core.coordinates.time import CoordTime

# from core.coordinates.trials import CoordError
from core.data_structures.base_data_struct import DataStructure
from core.entities.bio import Area, Training
from utils.io_data.formats import TargetType
from utils.storage_rulers.impl_path_rulers import FiringRatesPopPath
from utils.io_data.loaders.impl_loaders import LoaderPKL
from utils.io_data.savers.impl_savers import SaverPKL


class FiringRatesPop(DataStructure):
    """
    Firing rates for a pseudo-population in a set of pseudo-trials.

    Key Features
    ------------
    Dimensions : ``ensembles``, ``units``, ``folds``, ``trials``, ``time``

    Coordinates:

    - ``units`` (dimensions ``ensembles``, ``units``)
    - ``task``  (dimension ``trials``)
    - ``ctx``   (dimension ``trials``),
    - ``stim``  (dimension ``trials``),
    - `time``  (dimension ``time``)

    Identity Metadata: ``area``, ``training`

    Descriptive Metadata: ``error``

    Attributes
    ----------
    data : np.ndarray[Tuple[Any, Any, Any, Any, Any], np.float64]
        Firing rates time courses of all the units in all the trials.
        Shape: ``(n_ens, n_units, n_folds, n_trials, n_t)``, with:

        - ``n_ens``: Number of ensembles in the pseudo-population.
        - ``n_units``: Number of units in the pseudo-population.
        - ``n_folds``: Number of folds in the cross-validation scheme.
        - ``n_trials``: Number of pseudo-trials in the reconstructed data set.
        - ``n_t``: Number of time points in a trial's time course.

    units : np.ndarray[Tuple[Any, Any], np.str_]
        Labels of the units in each ensemble of the pseudo-population.
    task : np.ndarray[Tuple[Any], np.int64]
        Coordinate labels for the task from which each trial comes.
    ctx : np.ndarray[Tuple[Any], np.str_]
        Coordinate labels for the context from which each trial comes.
    stim : np.ndarray[Tuple[Any], np.str_]
        Coordinate labels for the stimulus presented in each trial.
    time : np.ndarray[Tuple[Any], np.float64]
        Time points of the firing rate time courses (in seconds).
    area : Area
        Brain area from which the units were recorded.
    training : Training
        Training condition of the animals from which the units were recorded.
    error : bool
        Whether the data set comprises error or valid trials.
    """

    # --- Schema Attributes ---
    dims = (
        DimName("ensembles"),
        DimName("units"),
        DimName("folds"),
        DimName("trials"),
        DimName("time"),
    )
    coords = MappingProxyType(
        {
            "units": CoordUnit,
            "task": CoordTask,
            "ctx": CoordCtx,
            "stim": CoordStim,
            "time": CoordTime,
        }
    )
    coords_to_dims = MappingProxyType(
        {
            "units": (DimName("ensembles"), DimName("units")),
            "task": (DimName("trials"),),
            "ctx": (DimName("trials"),),
            "stim": (DimName("trials"),),
            "time": (DimName("time"),),
        }
    )
    identifiers = ("area", "training")

    # --- IO Handlers ---
    path_ruler = FiringRatesPopPath
    loader = LoaderPKL
    saver = SaverPKL
    tpe = TargetType("ndarray_float")

    def __init__(
        self,
        area: Union[Area, str],
        training: Union[Training, str],
        error: bool = False,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        units: Optional[Union[CoordUnit, np.ndarray]] = None,
        task: Optional[Union[CoordTask, np.ndarray]] = None,
        ctx: Optional[Union[CoordCtx, np.ndarray]] = None,
        stim: Optional[Union[CoordStim, np.ndarray]] = None,
        time: Optional[Union[CoordTime, np.ndarray]] = None,
    ):
        # Set sub-class specific metadata
        self.area = Area(area)
        self.training = Training(training)
        self.error = error
        # Set data and coordinate attributes via the base class constructor
        super().__init__(
            data=data,
            units=units,
            task=task,
            ctx=ctx,
            stim=stim,
            time=time,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}>: Area {self.area}, Training {self.training}\n"
            + super().__repr__()
        )

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.area, self.training)
