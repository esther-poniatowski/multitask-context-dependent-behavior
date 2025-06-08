"""
`core.data_structures.events_properties` [module]

Classes
-------
EventsProperties
"""
from types import MappingProxyType

import numpy as np
import pandas as pd

from core.data_components.core_dimensions import Dimensions, DimensionsSpec
from core.data_components.base_data_component import ComponentSpec
from core.data_components.core_metadata import MetaDataField
from core.data_structures.base_data_structure import DataStructure
from core.data_components.core_data import CoreIndices
from core.coordinates.base_coordinate import Coordinate
from core.attributes.exp_structure import Session
from core.coordinates.exp_structure_coord import CoordBlock
from core.coordinates.time_coord import CoordTimeEvent
from core.coordinates.exp_factor_coord import CoordEventDescription


class EventsProperties(DataStructure[CoreIndices]):
    """
    Metadata about the events occurring in one recording session of the experiment (raw).

    Key Features
    ------------
    Dimensions : ``events``

    Coordinates:

    - ``block``
    - ``t_start``
    - ``t_end``
    - ``description``

    Identity Metadata: ``session``

    Attributes
    ----------
    data : CoreIndices
        Indices of the events in the session.
    block : CoordBlock
        Coordinate for the block of trials in which each event occurred.
    t_start, t_end : CoordTimeEvent
        Coordinates for the start and end times of each event.
    description : CoordEventDescription
        Coordinate for the nature of each event. Each element is a string comprising one or more
        event descriptions separated by commas. Examples: ``'PreStimSilence , TORC_448_06_v501 ,
        Reference'``, ``'TRIALSTART'``, etc.
    session : Session
        Session's identifier.

    Notes
    -----
    This data structure represents the entry point of the data analysis. Therefore, it tightly
    reflects to the raw data, without additional pre-processing. It is not meant to be saved,
    therefore no saver is defined.
    """

    # --- Data Structure Schema --------------------------------------------------------------------

    DIMENSIONS_SPEC = DimensionsSpec(events=True)
    COMPONENTS_SPEC = ComponentSpec(
        data=CoreIndices,
        block=CoordBlock,
        t_start=CoordTimeEvent,
        t_end=CoordTimeEvent,
        description=CoordEventDescription,
    )
    IDENTIFIERS = MappingProxyType({"session": MetaDataField(Session, "")})

    def __init__(
        self, session: Session, data: CoreIndices | None = None, **coords: Coordinate
    ) -> None:
        # Set sub-class specific metadata
        self.session = session
        # Set data and coordinate attributes via the base class constructor
        super().__init__(data=data, **coords)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: Session {self.session}\n" + super().__repr__()

    # --- Formatting -------------------------------------------------------------------------------

    # TODO: Define a mapping at the class level between CSV columns and the corresponding coordinates
    def format(
        self,
        raw: pd.DataFrame,
        columns=("TrialNum", "Event", "StartTime", "StopTime"),  # immutable
    ) -> None:
        """
        Format the raw data obtained as a pandas DataFrame from a CSV file.

        Parameters
        ----------
        raw : pd.DataFrame
            Raw data obtained from a CSV file.
        columns : List[str], default=['TrialNum', 'Event', 'StartTime', 'StopTime']
            Names of the columns expected in the CSV file.

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
        # Check the header of the CSV file
        if not np.array_equal(raw.columns, columns):
            raise ValueError(f"Invalid header in the raw data: {raw.columns} instead of {columns}")
        # Set dimensions
        n_events = raw.shape[0]
        dims = Dimensions("events")  # common dimension for data and coordinates
        # Extract data (events indices)
        data = CoreIndices(np.arange(n_events), dims=dims)
        # Extract coordinates
        block = CoordBlock(values=raw["TrialNum"].values.astype(np.int64), dims=dims)
        t_start = CoordTimeEvent(values=raw["StartTime"].values, dims=dims)
        t_end = CoordTimeEvent(values=raw["StopTime"].values, dims=dims)
        description = CoordEventDescription(values=raw["Event"].values, dims=dims)
        # Create new instance filled with the loaded data
        obj = EventsProperties(
            session=self.session,
            data=data,
            block=block,
            t_start=t_start,
            t_end=t_end,
            description=description,
        )
        self.__dict__.update(obj.__dict__)  # update the instance with the new data
