#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_session_trials` [module]

Classes
-------
`RasterBuilder`

Notes
-----
-
"""
# pylint: disable=missing-function-docstring

from typing import Optional

from core.builders.base_builder import DataStructureBuilder
from core.coordinates.base_coord import Coordinate
from core.coordinates.exp_factor_coord import CoordCategory
from core.coordinates.exp_structure_coord import CoordBlock, CoordSlot
from core.coordinates.time_coord import CoordTimeEvent
from core.coordinates.trials_coord import CoordError
from core.data_structures.core_data import CoreData
from core.data_structures.events_properties import EventsProperties
from core.data_structures.trials_properties import TrialsProperties


class TrialsPropertiesBuilder(DataStructureBuilder[EventsProperties, TrialsProperties]):
    """
    Build a `TrialsProperties` data structure from a `EventsProperties` data structure.

    Class Attributes
    ----------------
    product_class : type
        See the base class attribute.
    TMP_DATA : Tuple[str]
        See the base class attribute.

    Processing Attributes
    ---------------------
    session_events : EventsProperties
        Raw meta data about the events which occurred in the session.

    Methods
    -------
    `build` (implementation of the base class method)
    """

    product_class = TrialsProperties
    TMP_DATA = ("session_events",)

    def __init__(self) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters

        # Declare attributes to store inputs and intermediate results
        self.session_events: Optional[EventsProperties] = None

    def build(
        self,
        session_id: Optional[str] = None,
        session_events: Optional[EventsProperties] = None,
        **kwargs,
    ) -> TrialsProperties:
        """
        Implement the base class method.

        Parameters
        ----------
        session_id : str
            Identifier of the session.
        session_events : EventsProperties
            See the attribute `session_events`.

        Returns
        -------
        product : TrialsProperties
            Data structure product instance.
        """
        assert session_id is not None and session_events is not None
        # Store inputs
        self.session_events = session_events
        # Preliminary set up

        # Initialize the data structure with its metadata (base method)
        self.initialize_data_structure(session_id=session_id)
        # Add core data values to the data structure
        data = self.construct_core_data()
        self.add_data(data)
        # Add coordinates in the data structure
        block: CoordBlock
        slot: CoordSlot
        categ: CoordCategory
        t_on: CoordTimeEvent
        t_off: CoordTimeEvent
        t_warn: CoordTimeEvent
        t_end: CoordTimeEvent
        error: CoordError
        self.add_coords(
            block=block,
            slot=slot,
            categ=categ,
            t_on=t_on,
            t_off=t_off,
            t_warn=t_warn,
            t_end=t_end,
            error=error,
        )
        return self.get_product()

    # --- Shape and Dimensions ---------------------------------------------------------------------

    @property
    def n_trials(self) -> int:
        """Number of trials which occurred in the session."""
        return 0  # TODO

    # --- Preliminary Operations -------------------------------------------------------------------

    # --- Construct Core Data ----------------------------------------------------------------------

    def construct_core_data(self) -> CoreData:
        """
        Construct the core data array containing the spike trains of the unit in the trials.

        Returns
        -------
        data : CoreData
            Data array containing the spike times.
            Shape: ``(n_trials, n_spk_max)``.
            .. _data:

        Implementation
        --------------
        1. Initialize the core data array with empty values.
        2.

        See Also
        --------
        `DataStructureBuilder.initialize_core_data`
            Initialize the core data array with empty values (parent method).

        """
        shape = (self.n_trials,)
        data = self.initialize_core_data(shape)  # parent method
        # TODO
        return data

    # --- Construct Coordinates --------------------------------------------------------------------

    def construct_coord(self, name, coord_class: type[Coordinate]) -> Coordinate:
        """
        Construct a new coordinate  for the trials dimension containing session's information.

        Each session is associated with one recording number, one task and one attentional state, and yields
        as many number of elements as the total number of trials it contains.

        Parameters
        ----------
        name : str
            Name of the coordinate to build.
        coord_class : type[Coordinate]
            Class of the coordinate to build. Used to convert the array of values into a coordinate
            object of the appropriate type.

        Returns
        -------
        coord : Coordinate
            Coordinate for the trial dimension in the data structure product.
        """
        coord: Coordinate
        # TODO
        return coord
