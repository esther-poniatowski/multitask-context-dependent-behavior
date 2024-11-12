#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.trials_properties` [module]
"""
from types import MappingProxyType
from typing import Optional, Union, Generator, List

import numpy as np

from core.coordinates.exp_structure import CoordRecNum, CoordBlock, CoordSlot
from core.coordinates.exp_condition import CoordTask, CoordAttention, CoordStim
from core.coordinates.time import CoordTimeEvent
from core.coordinates.trials import CoordError
from core.data_structures.base_data_struct import DataStructure
from core.data_structures.core_data import Dimensions, CoreData
from core.entities.exp_structure import Session

# from utils.io_data.loaders import LoaderNPY
# from utils.storage_rulers.impl_path_rulers import TrialsPropertiesPath


class TrialsProperties(DataStructure):
    """
    Metadata about a set of trials the experiment.

    Base class which contains the common attributes and methods for both single session and
    multi-session trials properties.

    Key Features
    ------------
    Dimensions : ``trials``

    Coordinates:

    - ``recnum`` (optional)
    - ``block``
    - ``slot``
    - ``task`` (optional)
    - ``attn`` (optional)
    - ``stim``
    - ``t_on``
    - ``t_off``
    - ``t_warn``
    - ``t_end``
    - ``error``

    Attributes
    ----------
    session_ids : List[Session]
        Identifier(s) of the session(s) from which the trials come.
    data : CoreData
        Indices of the trials relative to the considered set (e.g. session or experiment).
    recnum : CoordRecNum
        Coordinate for the recording number of each trial (identifying a session at a site).
    block : CoordBlock
        Coordinate for the block of trials to which each trial belongs in its session.
    slot : CoordSlot
        Coordinate for the slot of each trial within its block.
    task : CoordTask, optional
        Coordinate for the task of each trial.
    attn : CoordContext, optional
        Coordinate for the attentional state of each trial.
    stim : CoordStim
        Coordinate for the nature of the stimulus presented in each trial.
    t_warn : CoordTimeEvent
        Coordinate for the onset of the warning sound (only in task CLK).
    t_on, t_off : CoordTimeEvent
        Coordinates for the onset and offset times of the stimulus in each trial.
    t_end : CoordTimeEvent
        Coordinate for the end time of each trial.
    error : CoordError
        Coordinate for the behavioral outcome of each trial.
    n_trials : int
        (Property) Number of trials in the subset.
    n_sessions : int
        (Property)
    n_blocks : int
        (Property)

    Methods
    -------
    `iter_trials`

    Notes
    -----
    This data structure can be used to reference trials in a single session or a full experiment
    (multiple sessions). The structure differs in both cases:

    For a single session:

    - The attribute `session_ids` contains a single session identifier.
    - The coordinates `recnum`, `task`, and `attn` are optional, since they would contain a unique
      label for all the trials.
    - The property `n_blocks` indicates the number of blocks in the unique session.

    For an experiment:

    - The attribute `session_ids` contains several session identifiers.
    - The coordinate `recnum` indicates the origin of each trial (i.e. the session from which it
      comes from).
    - The property `n_blocks` indicates the maximal number of blocks across the different sessions.

    Raises
    ------
    ValueError
        If the content of the coordinate `recnum` is not consistent with the sessions' IDs. This is
        the case if the unique labels contained in the coordinate do not match the attribute `rec`
        of the sessions' IDs.
    """

    # --- Schema Attributes ---
    dims = Dimensions("trials")
    coords = MappingProxyType(
        {
            "recnum": CoordRecNum,
            "block": CoordBlock,
            "slot": CoordSlot,
            "task": CoordTask,
            "attn": CoordAttention,
            "stim": CoordStim,
            "t_on": CoordTimeEvent,
            "t_off": CoordTimeEvent,
            "t_warn": CoordTimeEvent,
            "t_end": CoordTimeEvent,
            "error": CoordError,
        }
    )
    coords_to_dims = MappingProxyType({name: Dimensions("trials") for name in coords.keys()})
    identifiers = ("session_ids",)

    # --- IO Handlers ---
    # TODO

    # --- Key Features ---

    def __init__(
        self,
        session_ids: List[Session],
        data: Optional[Union[CoreData, np.ndarray]] = None,
        recnum: Optional[Union[CoordRecNum, np.ndarray]] = None,
        block: Optional[Union[CoordBlock, np.ndarray]] = None,
        slot: Optional[Union[CoordSlot, np.ndarray]] = None,
        task: Optional[Union[CoordTask, np.ndarray]] = None,
        attn: Optional[Union[CoordAttention, np.ndarray]] = None,
        stim: Optional[Union[CoordStim, np.ndarray]] = None,
        t_on: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_off: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_warn: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_end: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        error: Optional[Union[CoordError, np.ndarray]] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.session_ids = session_ids
        # Check recording numbers
        if recnum is not None:
            unique_labels = np.unique(recnum)
            sessions_recnums = [s.red for s in session_ids]
            if not all(label in sessions_recnums for label in unique_labels):
                raise ValueError(
                    "Invalid recording numbers in coordinate: "
                    f"{unique_labels} vs {sessions_recnums} in sessions' IDs."
                )
        # Set data and coordinate attributes via the base class constructor
        super().__init__(
            data=data,
            recnum=recnum,
            block=block,
            slot=slot,
            task=task,
            attn=attn,
            stim=stim,
            t_on=t_on,
            t_off=t_off,
            t_warn=t_warn,
            t_end=t_end,
            error=error,
        )

    @property
    def n_trials(self) -> int:
        """Number of trials in the subset."""
        return self.data.size

    @property
    def n_sessions(self) -> int:
        """Number of sessions from which the trials come from."""
        return len(self.session_ids)

    @property
    def n_blocks(self) -> int:
        """Maximal number of blocks across session(s)."""
        if len(self.block) != 0:  # avoid ValueError in max() if empty array
            return self.block.max()
        else:
            return 0

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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: Sessions {self.session_ids}\n" + super().__repr__()

    # --- IO Handling ----
