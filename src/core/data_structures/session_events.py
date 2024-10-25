#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.session_events` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Union

import numpy as np

from core.data_structures.base_data_struct import DataStructure
from core.data_structures.core_data import Dimensions, CoreData
from core.coordinates.exp_structure import CoordBlock
from core.coordinates.time import CoordTimeEvent
from core.coordinates.exp_condition import CoordEventDescription
from utils.io_data.formats import TargetType
from utils.io_data.loaders.impl_loaders import LoaderCSV
from utils.storage_rulers.impl_path_rulers import SessionEventsPath


class SessionEvents(DataStructure):
    """
    Metadata about one the events in one recording session of the experiment (raw).

    Key Features
    ------------
    Dimensions : ``events``

    Coordinates:

    - ``block``
    - ``t_start``
    - ``t_end``
    - ``description``

    Identity Metadata: ``session_id``

    Attributes
    ----------
    data : CoreData
        Indices of the events in the session.
    block : CoordBlock
        Coordinate for the block of trials in which each event occurred.
    t_start, t_end : CoordTimeEvent
        Coordinates for the start and end times of each event.
    description : CoordEventDescription
        Coordinate for the nature of the event. Each element is a string which can comprise several
        event descriptions separated by commas. Examples: ``'PreStimSilence , TORC_448_06_v501 ,
        Reference'``, ``'TRIALSTART'``, etc.
    session_id : str
        Session's identifier.

    Notes
    -----
    This data structure represents the entry point of the data analysis. Therefore, it tightly
    reflects to the raw data, without additional pre-processing. It is not meant to be saved,
    therefore no saver is defined.
    """

    # --- Schema Attributes ---
    dims = Dimensions("events")
    coords = MappingProxyType({})
    coords_to_dims = MappingProxyType({name: Dimensions("events") for name in coords.keys()})
    identifiers = ("session_id",)

    # --- IO Handlers ---
    path_ruler = SessionEventsPath
    loader = LoaderCSV
    tpe = TargetType("dataframe")

    # --- Key Features -----------------------------------------------------------------------------

    def __init__(
        self,
        session_id: str,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        block: Optional[Union[CoordBlock, np.ndarray]] = None,
        t_start: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        t_end: Optional[Union[CoordTimeEvent, np.ndarray]] = None,
        description: Optional[Union[CoordEventDescription, np.ndarray]] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.session_id = session_id
        # Set data and coordinate attributes via the base class constructor
        super().__init__(
            data=data, block=block, t_start=t_start, t_end=t_end, description=description
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: Session {self.session_id}\n" + super().__repr__()

    # --- IO Handling ------------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.session_id)

    def load(self) -> None:
        """
        Retrieve data from a CSV file and extract the coordinates.

        Notes
        -----
        The raw data of one session is the CSV output of a MATLAB script that parses ``.m`` files
        (``exptevents.m``).

        Rows in the CSV file:

        - Each row represents a single event from the experimental session.
        - The data is organized chronologically based on the order of events in the original
          'exptevents' structure.

        Columns in the CSV file:

        - ``TrialNum``: Block number in which the event occurred, extracted from the
          ``exptevents.Trial`` field.
        - ``Event``: Nature of the event, extracted from the ``exptevents.Note`` field.
        - ``StartTime``: Start time of the event (in seconds) relative to relative to the start of
          the session, extracted from the ``exptevents.StartTime`` field.
        - ``StopTime``: Stop time of the event (in seconds) relative to relative to the start of the
          session, extracted from the ``exptevents.StopTime`` field.

        Numeric values (TrialNum, StartTime, StopTime) are stored as numbers, while the Event
        description is stored as text.

        Header of the CSV file: ``{'TrialNum', 'Event', 'StartTime', 'StopTime'}``

        Examples
        --------
        Example of a row in the CSV file:

        .. code-block:: text

            TrialNum    Event                                           StartTime   StopTime
            1           'PreStimSilence , TORC_448_06_v501 , Reference'	0.0	        0.4

        Warning
        -------
        In the raw data, the term "trial" refers to a block of stimuli presentations.
        In the subsequent analysis, the term "trial" refers to a single slot form one block.
        """
        # Load numpy array via LoaderNPY
        raw = self.loader(path=self.path, tpe=self.tpe).load()  # dataframe
        # Check the header of the CSV file
        if not np.array_equal(raw.columns, ["TrialNum", "Event", "StartTime", "StopTime"]):
            raise ValueError(f"Invalid header in the CSV file at {self.path}.")
        # Build data (events indices)
        n_events = raw.shape[0]
        data = CoreData(np.arange(n_events), dims=("events",))
        # Extract coordinates
        block = CoordBlock(values=raw["TrialNum"].values.astype(np.int64))
        t_start = CoordTimeEvent(values=raw["StartTime"].values)
        t_end = CoordTimeEvent(values=raw["StopTime"].values)
        description = CoordEventDescription(values=raw["Event"].values)
        # Create new instance filled with the loaded data
        obj = SessionEvents(
            session_id=self.session_id,
            data=data,
            block=block,
            t_start=t_start,
            t_end=t_end,
            description=description,
        )
        self.__dict__.update(obj.__dict__)  # update the instance with the new data
