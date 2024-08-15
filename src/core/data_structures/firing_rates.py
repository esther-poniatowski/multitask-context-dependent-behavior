#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.firing_rates` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import numpy as np
import numpy.typing as npt


from core.constants import (
    T_BIN,
    T_MAX,
    T_ON,
    T_OFF,
    T_SHOCK,
    SMOOTH_WINDOW,
)
from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim
from core.coordinates.time import CoordTime
from core.coordinates.trials import CoordError
from core.data_structures.base import Data
from core.entities.bio import Area, Training
from utils.io.formats import TargetType
from utils.path_system.storage_rulers.impl import FiringRatesPopPath
from utils.io.loaders.impl import LoaderPKL
from utils.io.savers.impl import SaverPKL


class FiringRatesPop(Data):
    """
    Firing rates for a pseudo-population in a set of pseudo-trials.

    Key Features
    ------------
    Data       : ``data`` (type ``npt.NDArray[np.float64]``)
    Dimensions : ``units``, ``time``, ``trials``
    Coordinates: ``recnum`` (type ``CoordRecNum``)
                 ``block`` (type ``CoordBlock``)
                 ``slot`` (type ``CoordSlot``)
                 ``task`` (type ``CoordTask``)
                 ``ctx`` (type ``CoordCtx``)
                 ``stim`` (type ``CoordStim``)
                 ``error`` (type ``CoordError``)
    Metadata   : ``unit_id``, ``t_bin``, ``smooth_window``,
                 ``t_max``, ``t_on``, ``t_off``, ``t_shock``
                 ``n_units``, ``n_tpts``, ``n_trials``

    Attributes
    ----------
    data: npt.NDArray[np.float64]
        Firing rates time courses of all the units in all the trials.
        Shape: ``(n_units, n_t, n_trials)``.
    n_units: int
        Total number of units in the pseudo-population.
    n_trials: int
        Total number of pseudo-trials in the reconstructed data set.
    n_t: int
        Number of time points in a trial's time course.
    time: CoordTime
        Coordinate for dimension `time`.
    task: CoordTask
        Coordinate for dimension `trials`.
    ctx: CoordCtx
        Coordinate for dimension `trials`.
    stim: CoordStim
        Coordinate for dimension `trials`.
    error: CoordError
        Coordinate for dimension `trials`.
    t_max: float
        Total duration the firing rate time course (in seconds), homogeneous across trials.
    t_on, t_off: float
        Times of the stimulus onset and offset (in seconds), homogeneous across trials.
    t_bin: float
        Time bin for the firing rate time course (in seconds).
    smooth_window: float
        Smoothing window size (in seconds).
    area: Area
        Brain area from which the units were recorded.
    training: Training
        Training condition of the animals from which the units were recorded.
    """

    dim2coord = MappingProxyType({"trial": frozenset(["task", "ctx", "stim", "error"])})
    coord2type = MappingProxyType(
        {
            "time": CoordTime,
            "task": CoordTask,
            "ctx": CoordCtx,
            "stim": CoordStim,
            "error": CoordError,
        }
    )
    path_ruler = FiringRatesPopPath
    loader = LoaderPKL
    saver = SaverPKL
    tpe = TargetType("ndarray_float")

    def __init__(
        self,
        area: Area,
        training: Training,
        t_bin: float = T_BIN,
        t_max: float = T_MAX,
        t_on: float = T_ON,
        t_off: float = T_OFF,
        t_shock: float = T_SHOCK,
        smooth_window: float = SMOOTH_WINDOW,
        data: Optional[npt.NDArray[np.float64]] = None,
        time: Optional[CoordTime] = None,
        task: Optional[CoordTask] = None,
        ctx: Optional[CoordCtx] = None,
        stim: Optional[CoordStim] = None,
        error: Optional[CoordError] = None,
    ):
        # Set sub-class specific metadata
        self.area = area
        self.training = training
        self.t_bin = t_bin
        self.t_max = t_max
        self.t_on = t_on
        self.t_off = t_off
        self.t_shock = t_shock
        self.smooth_window = smooth_window
        # Declare data and coordinate attributes (avoid type errors)
        self.data: npt.NDArray[np.float64]
        self.task: CoordTask
        self.ctx: CoordCtx
        self.stim: CoordStim
        self.error: CoordError
        # Set data and coordinate attributes
        super().__init__(
            data=data,
            time=time,
            task=task,
            ctx=ctx,
            stim=stim,
            error=error,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}>: Area {self.area}, Training {self.training}\n"
            + super().__repr__()
        )

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.area.value, self.training.value)

    @property
    def n_units(self) -> int:
        return self.data.shape[0]

    @property
    def n_t(self) -> int:
        return self.data.shape[1]

    @property
    def n_trials(self) -> int:
        return self.data.shape[2]
