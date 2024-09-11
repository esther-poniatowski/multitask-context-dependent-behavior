#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.trials_properties` [module]
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
from core.data_structures.base import Data
from utils.storage_rulers.impl import TrialsPropertiesPath
from utils.io_data.loaders.impl import LoaderPKL
from utils.io_data.savers.impl import SaverPKL


class TrialsProperties(Data):
    """
    Properties of a set of trials from one session or several sessions recorded at the same site.

    Key Features
    ------------
    Data       : ``data`` (type ``npt.NDArray[np.int64]``)
    Dimensions : ``trials``
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
    Metadata   : ``smpl_rate``, ``site_id``, ``sessions_ids``

    Attributes
    ----------
    data: npt.NDArray[np.int64]
        Trial identifiers. Shape: ``(n_trials, )``
        Boundaries: From 1 to the total number of trials across all the sessions recorded at one
        site (to ensure unique identifiers and consistency when merging sessions).
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
    smpl_rate: float
        Sampling time for the recording (in seconds).
        Default: :obj:`core.constants.SMPL_RATE`
    site_id: str
        Identifier of the recording site.
    sessions_ids: List[str]
        Identifiers of the session(s) used to build the data structure.

    See Also
    --------
    :class:`core.coordinates.exp_structure.CoordRecNum`
    :class:`core.coordinates.exp_structure.CoordBlock`
    :class:`core.coordinates.exp_structure.CoordSlot`
    :class:`core.coordinates.exp_condition.CoordTask`
    :class:`core.coordinates.exp_condition.CoordCtx`
    :class:`core.coordinates.exp_condition.CoordStim`
    :class:`core.coordinates.trials.CoordError`
    :class:`core.
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
    path_ruler = TrialsPropertiesPath
    loader = LoaderPKL
    saver = SaverPKL

    def __init__(
        self,
        site_id: str,
        sessions_ids: list[str],
        smpl_rate: float = SMPL_RATE,
        data: Optional[npt.NDArray[np.int64]] = None,
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
        self.site_id = site_id
        self.sessions_ids = sessions_ids
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
        return f"<{self.__class__.__name__}>: Site {self.site_id}\n" + super().__repr__()

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.site_id)
