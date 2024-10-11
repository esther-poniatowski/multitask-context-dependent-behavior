#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.metadata_session` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Union, Generator

import numpy as np

from core.coordinates.exp_structure import CoordBlock, CoordSlot
from core.coordinates.exp_condition import CoordStim
from core.coordinates.time import CoordTimeEvent
from core.coordinates.trials import CoordError
from core.data_structures.base_data_struct import DataStructure
from core.data_structures.core_data import Dimensions, CoreData
from utils.io_data.formats import TargetType
from utils.io_data.loaders.impl_loaders import LoaderNPY
from utils.storage_rulers.impl_path_rulers import SessionTrialsPath


class SessionTrials(DataStructure):
    """
    Metadata about the trials in one recording session of the experiment.

    Key Features
    ------------
    Dimensions : ``trials``

    Coordinates:

    - ``block``
    - ``slot``
    - ``stim``
    - ``t_on``
    - ``t_off``
    - ``t_warn``
    - ``t_end``
    - ``error``

    Identity Metadata: ``session_id``

    Attributes
    ----------
    data: CoreData
        Indices of the trials in the session.
    block: CoordBlock
        Coordinate for the block of trials from which each trial comes from.
    slot: CoordSlot
        Coordinate for the slot of each trial within its block.
    stim: CoordStim
        Coordinate for the nature of the stimulus presented in each trial.
    t_warn: CoordTimeEvent
        Coordinate for the onset of the warning sound (only in task CLK).
    t_on, t_off: CoordTimeEvent
        Coordinates for the onset and offset times of the stimulus in each trial.
    t_end: CoordTimeEvent
        Coordinate for the end time of each trial.
    session_id: str
        Session's identifier.

    Methods
    -------
    `iter_trials`
    """

    # --- Schema Attributes ---
    dims = Dimensions("trials")
    coords = MappingProxyType(
        {
            "block": CoordBlock,
            "slot": CoordSlot,
            "stim": CoordStim,
            "t_on": CoordTimeEvent,
            "t_off": CoordTimeEvent,
            "t_warn": CoordTimeEvent,
            "t_end": CoordTimeEvent,
            "error": CoordError,
        }
    )
    coords_to_dims = MappingProxyType({name: Dimensions("trials") for name in coords.keys()})
    identifiers = ("session_id",)

    # --- IO Handlers ---
    path_ruler = SessionTrialsPath
    loader = LoaderNPY
    tpe = TargetType("ndarray_float")

    # --- Key Features -----------------------------------------------------------------------------

    def __init__(
        self,
        session_id: str,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        block: Optional[Union[CoordBlock, np.ndarray]] = None,
        slot: Optional[Union[CoordSlot, np.ndarray]] = None,
        stim: Optional[Union[CoordStim, np.ndarray]] = None,
        t_on: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_off: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_warn: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_end: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.session_id = session_id
        # Set data and coordinate attributes via the base class constructor
        super().__init__(
            data=data,
            block=block,
            slot=slot,
            stim=stim,
            t_on=t_on,
            t_off=t_off,
            t_warn=t_warn,
            t_end=t_end,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: Session {self.session_id}\n" + super().__repr__()

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the session."""
        if len(self.block) != 0:  # avoid ValueError in max() if empty array
            return self.block.max()
        else:
            return 0

    @property
    def n_trials(self) -> int:
        """Number of trials in the session."""
        return self.data.size

    def iter_trials(self) -> Generator:
        """
        Iterate over the trials in the session and yield their metadata.

        Yields
        ------
        block : int
            Block number of the trial.
        slot : int
            Slot number of the trial within its block.
        t_start : float
            Start time of the trial (onset of the stimulus).
        t_end : float
            End time of the trial.
        """
        for block, slot, t_on, t_end in zip(self.block, self.slot, self.t_on, self.t_end):
            yield block, slot, t_on, t_end

    # --- IO Handling ------------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.session_id)
