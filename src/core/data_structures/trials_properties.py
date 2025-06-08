"""
`core.data_structures.trials_properties` [module]

Classes
-------
TrialsProperties
"""
from types import MappingProxyType
from typing import Generator, List, Self

import numpy as np

from core.data_components.core_dimensions import DimensionsSpec
from core.data_components.base_data_component import ComponentSpec
from core.data_components.core_metadata import MetaDataField
from core.data_components.core_data import CoreIndices
from core.coordinates.base_coordinate import Coordinate
from core.coordinates.exp_structure_coord import CoordRecording, CoordBlock, CoordSlot
from core.coordinates.exp_factor_coord import (
    CoordTask,
    CoordAttention,
    CoordCategory,
    CoordBehavior,
    CoordOutcome,
)
from core.coordinates.time_coord import CoordTimeEvent
from core.data_structures.base_data_structure import DataStructure
from core.attributes.exp_structure import Session, Recording


class TrialsProperties(DataStructure[CoreIndices]):
    """
    Metadata about a set of trials the experiment (from a single or multiple sessions).

    Key Features
    ------------
    Dimensions : ``trials``

    Coordinates:

    - ``recording`` (optional)
    - ``block``
    - ``slot``
    - ``task``      (optional)
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
    data : CoreIndices
        Indices of the trials relative to the considered set (e.g. session or experiment).
    recording : CoordRecording
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
        Coordinate for the behavioral choice of the animal in each trial.
    outcome : CoordOutcome
        Coordinate for the outcome of each trial (correct, incorrect, etc.).
    t_warn : CoordTimeEvent
        Coordinate for the onset of the warning sound (only in task CLK).
    t_on, t_off : CoordTimeEvent
        Coordinates for the onset and offset times of the stimulus in each trial.
    t_start : CoordTimeEvent
        Coordinate for the start time of each trial relative to its block.
    t_end : CoordTimeEvent
        Coordinate for the end time of each trial relative to its block.
    t_dur : CoordTimeEvent
        Coordinate for the duration of each trial.
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
    get_subset
    get_session
    get_info
    iter_trials
    map_recording_to_session

    Notes
    -----
    This data structure can be used to reference trials in a single session or a full experiment
    (multiple sessions). The structure differs in both cases:

    For a single session:

    - The attribute `sessions` contains a single session identifier.
    - The coordinates `recording`, `task`, and `attention` are optional, since they would contain a
      unique label for all the trials.
    - The property `n_blocks` indicates the number of blocks in the unique session.

    For an experiment:

    - The attribute `sessions` contains several session identifiers.
    - The coordinate `recording` indicates the origin of each trial (i.e. the session from which it
      comes from).
    - The property `n_blocks` indicates the maximal number of blocks across the different sessions.

    Raises
    ------
    ValueError
        If the content of coordinates (`recording`, `task`, `attention` is not consistent with the
        sessions' IDs. This is the case if the unique labels contained in the coordinate do not
        match the respective values in the sessions' IDs.
    """

    # --- Data Structure Schema --------------------------------------------------------------------

    DIMENSIONS_SPEC = DimensionsSpec(trials=True)
    COMPONENTS_SPEC = ComponentSpec(
        data=CoreIndices,
        recording=CoordRecording,
        block=CoordBlock,
        slot=CoordSlot,
        task=CoordTask,
        attention=CoordAttention,
        category=CoordCategory,
        behavior=CoordBehavior,
        outcome=CoordOutcome,
        t_on=CoordTimeEvent,
        t_off=CoordTimeEvent,
        t_warn=CoordTimeEvent,
        t_end=CoordTimeEvent,
    )
    IDENTIFIERS = MappingProxyType({"sessions": MetaDataField(list, [])})

    def __init__(
        self, sessions: List[Session], data: CoreIndices | None = None, **coords: Coordinate
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

    # --- Getter Methods ---------------------------------------------------------------------------

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

    def get_subset(self, idx: np.ndarray | List[int]) -> Self:
        """
        Get a subset of the trials based on a boolean mask.

        Parameters
        ----------
        idx : ArrayLike
            Boolean mask to select the trials.

        Returns
        -------
        subset : TrialsProperties
            Subset of the trials corresponding to the mask.
        """
        data = self.data[idx]
        coords = {name: self.get_coord(name)[idx] for name in self.coords}
        return self.__class__(sessions=self.sessions, data=data, **coords)

    def get_session(self, session: str | Session) -> Self:
        """
        Get a subset of the trials corresponding to a single session.

        Parameters
        ----------
        session : str or Session
            Identifier of the session to select.

        Returns
        -------
        subset : TrialsProperties
            Subset of the trials corresponding to the session.

        Raises
        ------
        ValueError
            If the session identifier is not found in the sessions' IDs.

        See Also
        --------
        `core.attributes.exp_structure.Session`
        """
        # Validate the session identifier
        if isinstance(session, str):
            session = Session(session)
        if session not in self.sessions:
            raise ValueError(f"Invalid session identifier ({session}) in sessions: {self.sessions}")
        # Find the trials' indices from the recording number of the session
        recording = session.recording
        coord = self.get_coord("recording")
        idx = coord == recording
        # Return the subset of trials
        return self.get_subset(idx)

    def get_info(self, trial: int, *args: str) -> tuple:
        """
        Get the metadata of a trial.

        Parameters
        ----------
        trial : int
            Index of the trial in the subset.
        args : str
            Names of the coordinates to return.

        Returns
        -------
        info : Tuple
            Metadata of the trial, in the order of the requested coordinates.

        Examples
        --------
        >>> trial_info = trials.get_info(0, "block", "category")
        >>> print(trial_info)
        (1, "R")
        """
        return tuple(self.get_coord(name)[trial] for name in args)

    def iter_trials(self, *args: str) -> Generator[tuple, None, None]:
        """
        Iterate over the trials in the session and yield their metadata.

        Arguments
        ---------
        args : str
            Names of the coordinates to yield.

        Yields
        ------
        trial : Tuple
            Metadata of the trial, in the order of the requested coordinates.
        """
        for i in range(self.n_trials):
            yield tuple(self.get_coord(name)[i] for name in args)

    def map_recording_to_session(self, recording: int | Recording) -> Session:
        """
        Provide the session identifier corresponding to one recording number.

        Parameters
        ----------
        recording : int or Recording
            Recording number.

        Returns
        -------
        session : Session
            Identifier of the session from which the trial comes.

        Raises
        ------
        ValueError
            If the recording number is not found in the sessions' IDs.

        See Also
        --------
        `core.attributes.exp_structure.Session.recording`
        """
        if isinstance(recording, int):
            recording = Recording(recording)
        recordings_to_sessions = {s.recording: s for s in self.sessions}
        if recording not in recordings_to_sessions:
            raise ValueError(f"Invalid recording number ({recording}) in sessions: {self.sessions}")
        return recordings_to_sessions[recording]
