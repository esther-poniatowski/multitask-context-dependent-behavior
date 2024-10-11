#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.spikes_trains` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Union

import numpy as np

from core.constants import SMPL_RATE
from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim
from core.coordinates.exp_structure import CoordRecNum, CoordBlock, CoordSlot
from core.coordinates.time import CoordTimeEvent
from core.coordinates.trials import CoordError
from core.data_structures.base_data_struct import DataStructure
from core.data_structures.core_data import Dimensions, CoreData
from utils.io_data.formats import TargetType
from utils.storage_rulers.impl_path_rulers import SpikeTrainsPath
from utils.io_data.loaders.impl_loaders import LoaderPKL
from utils.io_data.savers.impl_savers import SaverPKL


class SpikeTrains(DataStructure):
    """
    Spikes trains for one unit in a set of trials of the experiment (whole or part).

    Key Features
    ------------
    Dimensions : ``trials``, ``time``

    Coordinates:

    - ``recnum`` (dimension ``trials``)
    - ``block``  (dimension ``trials``)
    - ``slot``   (dimension ``trials``)
    - ``task``   (dimension ``trials``)
    - ``ctx``    (dimension ``trials``)
    - ``stim``   (dimension ``trials``)
    - ``error``  (dimension ``trials``)
    - ``t_on``   (dimension ``trials``)
    - ``t_off``  (dimension ``trials``)
    - ``t_end``  (dimension ``trials``)

    Identity Metadata: ``unit_id``

    Descriptive Metadata: ``smpl_rate``

    Attributes
    ----------
    data: CoreData
        Spiking times of the unit in each trial (in seconds).
        Shape: ``(n_trials, n_spk_max)``, with ``n_spk_max`` the maximal number of spikes across all
        the trials of the unit.
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
        Coordinate for dimension `trials`, container for the `Trials.t_on` attribute.
    t_off: npt.ndarray[Tuple[Any], np.float64]
        Coordinate for dimension `trials`, container for the `Trials.t_off` attribute.
    t_end: npt.NDArray[float]
        Coordinate for dimension `trials`, container for the `Trials.t_end` attribute.
        Usually longer than the time of the last spike.
    unit_id: str
        Unit's identifier.
    smpl_rate: float, default=:obj:`core.constants.SMPL_RATE`
        Sampling time for the recording (in seconds).

    Warning
    -------
    In each trial, the spiking times are reset at 0 relative to the beginning of the *slot*,
    instead of the beginning of the block.

    Notes
    -----
    The data is stored in a *padded* array, where the shape ``n_spk_max`` is determined by the
    maximal number of spikes across trials. If the ith trial contains ``n_spk`` spikes, then the row
    ``data[i]`` associated to this trial has :

    - ``n_spk`` cells storing the spiking times (from index 0 to ``n_spk-1``)
    - ``n_spk_max - n_spk`` remaining cells filled with ``NaN``.

    This data structure represents an intermediary step between the raw data and the pre-processed
    data that will be analyzed. It centralizes information about spikes and trials to avoid
    repeating expensive computations. Indeed, most coordinates have to be built by parsing the raw
    ``expt_events`` files of the sessions.

    This data structure can be used for the following purposes :

    - Visualizing raster plots before firing rate are computed.
    - Selecting trials and units for the final analyses.

    See Also
    --------
    `core.coordinates.exp_structure.CoordRecNum`
    `core.coordinates.exp_structure.CoordBlock`
    `core.coordinates.exp_structure.CoordSlot`
    `core.coordinates.exp_condition.CoordTask`
    `core.coordinates.exp_condition.CoordCtx`
    `core.coordinates.exp_condition.CoordStim`
    `core.coordinates.trials.CoordError`
    """

    # --- Schema Attributes ---
    dims = Dimensions("trials", "time")
    coords = MappingProxyType(
        {
            "recnum": CoordRecNum,
            "block": CoordBlock,
            "slot": CoordSlot,
            "task": CoordTask,
            "ctx": CoordCtx,
            "stim": CoordStim,
            "error": CoordError,
            "t_on": CoordTimeEvent,
            "t_off": CoordTimeEvent,
            "t_end": CoordTimeEvent,
        }
    )
    coords_to_dims = MappingProxyType(
        {
            "recnum": Dimensions("trials"),
            "block": Dimensions("trials"),
            "slot": Dimensions("trials"),
            "task": Dimensions("trials"),
            "ctx": Dimensions("trials"),
            "stim": Dimensions("trials"),
            "error": Dimensions("trials"),
            "t_on": Dimensions("trials"),
            "t_off": Dimensions("trials"),
            "t_end": Dimensions("trials"),
        }
    )
    identifiers = ("unit_id",)

    # --- IO Handlers ---
    path_ruler = SpikeTrainsPath
    loader = LoaderPKL
    saver = SaverPKL
    tpe = TargetType("ndarray_float")

    def __init__(
        self,
        unit_id: str,
        smpl_rate: float = SMPL_RATE,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        recnum: Optional[Union[CoordRecNum, np.ndarray]] = None,
        block: Optional[Union[CoordBlock, np.ndarray]] = None,
        slot: Optional[Union[CoordSlot, np.ndarray]] = None,
        task: Optional[Union[CoordTask, np.ndarray]] = None,
        ctx: Optional[Union[CoordCtx, np.ndarray]] = None,
        stim: Optional[Union[CoordStim, np.ndarray]] = None,
        error: Optional[Union[CoordError, np.ndarray]] = None,
        t_on: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_off: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_end: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.unit_id = unit_id
        self.smpl_rate = smpl_rate
        # Set data and coordinate attributes via the base class constructor
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
