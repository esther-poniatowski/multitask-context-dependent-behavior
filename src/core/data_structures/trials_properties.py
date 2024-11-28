#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.trials_properties` [module]
"""
from types import MappingProxyType
from typing import Generator, List

from core.coordinates.exp_structure_coord import CoordRecNum, CoordBlock, CoordSlot
from core.coordinates.exp_factor_coord import (
    Coordinate,
    CoordTask,
    CoordAttention,
    CoordCategory,
    CoordBehavior,
    CoordOutcome,
)
from core.coordinates.time_coord import CoordTimeEvent
from core.data_structures.base_data_struct import DataStructure
from core.data_structures.core_data import Dimensions, CoreData
from core.attributes.exp_structure import Session, Recording


class TrialsProperties(DataStructure):
    """
    Metadata about a set of trials the experiment (from a single or multiple sessions).

    Key Features
    ------------
    Dimensions : ``trials``

    Coordinates:

    - ``recnum`` (optional)
    - ``block``
    - ``slot``
    - ``task`` (optional)
    - ``attention`` (optional)
    - ``category``
    - ``behavior``
    - ``outcome```
    - ``t_on``
    - ``t_off``
    - ``t_warn``
    - ``t_end``

    Metadata: ``sessions``

    Attributes
    ----------
    sessions : List[Session]
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
    attention : CoordAttention, optional
        Coordinate for the attentional state of each trial.
    category : CoordCategory
        Coordinate for the nature of the stimulus presented in each trial.
    behavior : CoordBehavior
        Coordinate for the behavioral choice of each trial.
    outcome : CoordOutcome
        Coordinate for the outcome of each trial.
    t_warn : CoordTimeEvent
        Coordinate for the onset of the warning sound (only in task CLK).
    t_on, t_off : CoordTimeEvent
        Coordinates for the onset and offset times of the stimulus in each trial.
    t_end : CoordTimeEvent
        Coordinate for the end time of each trial.
    error : CoordError
        Coordinate for the behavioral choice of each trial.
    n_trials : int
        (Property) Number of trials in the subset.
    n_sessions : int
        (Property) Number of sessions from which the trials come.
    n_blocks : int
        (Property) Maximal number of blocks across session(s).

    Methods
    -------
    `iter_trials`
    `get_session_from_recording`

    Notes
    -----
    This data structure can be used to reference trials in a single session or a full experiment
    (multiple sessions). The structure differs in both cases:

    For a single session:

    - The attribute `sessions` contains a single session identifier.
    - The coordinates `recnum`, `task`, and `attention` are optional, since they would contain a
      unique label for all the trials.
    - The property `n_blocks` indicates the number of blocks in the unique session.

    For an experiment:

    - The attribute `sessions` contains several session identifiers.
    - The coordinate `recnum` indicates the origin of each trial (i.e. the session from which it
      comes from).
    - The property `n_blocks` indicates the maximal number of blocks across the different sessions.


    TODO: Add a method to check the consistency of the coordinates with the sessions' IDs.
    Raises
    ------
    ValueError
        If the content of coordinates (`recnum`, `task`, `attention` is not consistent with the
        sessions' IDs. This is the case if the unique labels contained in the coordinate do not
        match the respective values in the sessions' IDs.
    """

    # --- Schema Attributes ---
    dims = Dimensions("trials")
    coords = MappingProxyType(
        {
            "recnum": CoordRecNum,
            "block": CoordBlock,
            "slot": CoordSlot,
            "task": CoordTask,
            "attention": CoordAttention,
            "category": CoordCategory,
            "behavior": CoordBehavior,
            "outcome": CoordOutcome,
            "t_on": CoordTimeEvent,
            "t_off": CoordTimeEvent,
            "t_warn": CoordTimeEvent,
            "t_end": CoordTimeEvent,
        }
    )
    coords_to_dims = MappingProxyType({name: Dimensions("trials") for name in coords.keys()})
    identifiers = ("sessions",)

    # --- Key Features ---

    def __init__(
        self,
        sessions: List[Session],
        data: CoreData | None = None,
        **coords: Coordinate,
    ) -> None:
        # Set sub-class specific metadata
        self.sessions = sessions
        # Set data and coordinate attributes via the base class constructor
        super().__init__(data=data, **coords)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}>: Sessions {self.sessions}, "
            f"#trials={self.n_trials}" + super().__repr__()
        )

    @property
    def n_trials(self) -> int:
        """Number of trials in the subset (length of the 1D core data)."""
        return self.data.size

    @property
    def n_sessions(self) -> int:
        """Number of sessions from which the trials come from."""
        return len(self.sessions)

    @property
    def n_blocks(self) -> int:
        """Maximal number of blocks across session(s)."""
        coord_blocks = self.get_coord("block")
        if len(coord_blocks) != 0:  # avoid ValueError in max() if empty array
            return coord_blocks.max()
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
        for block, slot, t_on, t_end in zip(
            self.get_coord("block"),
            self.get_coord("slot"),
            self.get_coord("t_on"),
            self.get_coord("t_end"),
        ):
            yield block, slot, t_on, t_end

    def get_session_from_recording(self, recnum: int | Recording) -> Session:
        """
        Get the session identifier corresponding to one recording number.

        Parameters
        ----------
        recnum : int or Recording
            Recording number.

        Returns
        -------
        session : Session
            Identifier of the session from which the trial comes.

        Raises
        ------
        ValueError
            If the recording number is not found in the sessions' IDs.

        return next((s for s, rec in sessions_to_recnums.items() if rec == recnum), None)
        --------
        `core.attributes.exp_structure.Session.rec`
        """
        if isinstance(recnum, int):
            recnum = Recording(recnum)
        recnums_to_sessions = {s.rec: s for s in self.sessions}
        if recnum not in recnums_to_sessions:
            raise ValueError(f"Invalid recording number ({recnum}) in sessions: {self.sessions}")
        return recnums_to_sessions[recnum]
