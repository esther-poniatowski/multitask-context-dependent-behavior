#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.spikes_trains` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import numpy as np
import numpy.typing as npt

from core.constants import SMPL_RATE
from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim
from core.coordinates.exp_structure import CoordRecNum, CoordBlock, CoordSlot
from core.coordinates.trials import CoordError
from core.data_structures.base import DataStructure
from utils.io_data.formats import TargetType
from utils.storage_rulers.impl import SpikesTrainsPath
from utils.io_data.loaders.impl import LoaderPKL
from utils.io_data.savers.impl import SaverPKL


class SpikesTrains(DataStructure):
    """
    Spikes trains for one unit in a set of trials of the experiment (whole or part).

    Key Features
    ------------
    Data       : ``data`` (type ``npt.NDArray[np.float64]``)
    Dimensions : ``time``, ``trials``
    Coordinates: ``recnum`` (type ``CoordRecNum``)
                 ``block`` (type ``CoordBlock``)
                 ``slot`` (type ``CoordSlot``)
                 ``task`` (type ``CoordTask``)
                 ``ctx`` (type ``CoordCtx``)
                 ``stim`` (type ``CoordStim``)
                 ``error`` (type ``CoordError``)
                 ``t_on`` (type ``npt.NDArray[np.float64]``)
                 ``t_off`` (type ``npt.NDArray[np.float64]``)
                 ``t_end`` (type ``npt.NDArray[np.float64]``)
    Metadata   : ``unit_id``, ``smpl_rate``

    Attributes
    ----------
    data: npt.NDArray[np.float64]
        Spiking times of the unit in each trial (in seconds).
        Shape: ``(n_trials, n_spk_max)``
        with ``n_spk_max`` the maximal number of spikes across trials.
    rec: CoordRecNum
        Coordinate for dimension `trials`.
    block: CoordBlock
        Coordinate for dimension `trials`.
    slot: CoordSlot
        Coordinate for dimension `trials`.
    task: CoordTask
        Coordinate for dimension `trials`.
    ctx: CoordCtx
        Coordinate for dimension `trials`.
    stim: CoordStim
        Coordinate for dimension `trials`.
    error: CoordError
        Coordinate for dimension `trials`.
    t_on: npt.NDArray[float]
        Coordinate for dimension `trials`, container for :attr:`t_on` in :class:`Trials`.
    t_off: npt.NDArray[float]
        Coordinate for dimension `trials`, container for :attr:`t_off` in :class:`Trials`.
    t_end: npt.NDArray[float]
        Coordinate for dimension `trials`, container for :attr:`t_end` in :class:`Trials`.
        Usually longer than the time of the last spike.
    unit_id: str
        Unit's identifier.
    smpl_rate: float
        Sampling time for the recording (in seconds).
        Default: :obj:`core.constants.SMPL_RATE`

    Warning
    -------
    In each trial, the spiking times are reset at 0 relative to the beginning of the *slot*,
    instead of the beginning of the block.

    Notes
    -----
    The shape ``n_spk_max`` of the data array is determined by the maximal number of spikes across trials.
    If the ith trial contains ``n_spk`` spikes, then the row ``data[i]`` associated to this trial has :

    - ``n_spk`` cells storing the spiking times (from index 0 to ``n_spk-1``)
    - ``n_spk_max - n_spk`` remaining cells filled with ``NaN``.

    This data structure represents an intermediary step between :class:`RawSpkTimes` and :class:`FiringRates`.
    It centralizes information about spikes and trials to avoid repeating expensive computations.
    Indeed, most coordinates have to be built by parsing the raw ``expt_events`` files of the sessions.

    This data structure can be used for the following purposes :

    - Visualizing raster plots before firing rate are computed.
    - Selecting trials and units for the final analyses.
      To assess several criteria, additional processing is required on each separate trial.
      New coordinates can be added to store the filtering metrics.

    See Also
    --------
    :class:`core.coordinates.exp_structure.CoordRecNum`
    :class:`core.coordinates.exp_structure.CoordBlock`
    :class:`core.coordinates.exp_structure.CoordSlot`
    :class:`core.coordinates.exp_condition.CoordTask`
    :class:`core.coordinates.exp_condition.CoordCtx`
    :class:`core.coordinates.exp_condition.CoordStim`
    :class:`core.coordinates.trials.CoordError`
    """

    dim2coord = MappingProxyType(
        {"trial": frozenset(["recnum", "block", "slot", "task", "ctx", "stim", "error"])}
    )
    coord2type = MappingProxyType(
        {
            "recnum": CoordRecNum,
            "block": CoordBlock,
            "slot": CoordSlot,
            "task": CoordTask,
            "ctx": CoordCtx,
            "stim": CoordStim,
            "error": CoordError,
            "t_on": npt.NDArray[np.float64],
            "t_off": npt.NDArray[np.float64],
            "t_end": npt.NDArray[np.float64],
        }
    )
    path_ruler = SpikesTrainsPath
    loader = LoaderPKL
    saver = SaverPKL
    tpe = TargetType("ndarray_float")

    def __init__(
        self,
        unit_id: str,
        smpl_rate: float = SMPL_RATE,
        data: Optional[npt.NDArray[np.float64]] = None,
        recnum: Optional[CoordRecNum] = None,
        block: Optional[CoordBlock] = None,
        slot: Optional[CoordSlot] = None,
        task: Optional[CoordTask] = None,
        ctx: Optional[CoordCtx] = None,
        stim: Optional[CoordStim] = None,
        error: Optional[CoordError] = None,
        t_on: Optional[npt.NDArray[np.float64]] = None,
        t_off: Optional[npt.NDArray[np.float64]] = None,
        t_end: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.unit_id = unit_id
        self.smpl_rate = smpl_rate
        # Declare data and coordinate attributes (avoid type errors)
        self.data: npt.NDArray[np.float64]
        self.recnum: CoordRecNum
        self.block: CoordBlock
        self.slot: CoordSlot
        self.task: CoordTask
        self.ctx: CoordCtx
        self.stim: CoordStim
        self.error: CoordError
        self.t_on: npt.NDArray[np.float64]
        self.t_off: npt.NDArray[np.float64]
        self.t_end: npt.NDArray[np.float64]
        # Set data and coordinate attributes
        super().__init__(
            data=data,
            recnum=recnum,
            block=block,
            slot=slot,
            task=task,
            ctx=ctx,
            stim=stim,
            error=error,
            t_on=t_on,
            t_off=t_off,
            t_end=t_end,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: Unit {self.unit_id}\n" + super().__repr__()

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.unit_id)
