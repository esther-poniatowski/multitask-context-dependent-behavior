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
from utils.io_data.formats import TargetType
from utils.io_data.loaders.impl_loaders import LoaderNPY
from utils.storage_rulers.impl_path_rulers import SessionEventsPath


class SessionEvents(DataStructure):
    """
    Metadata about one the events in one recording session of the experiment (raw).

    Key Features
    ------------
    Dimensions : ``events``

    Coordinates:

    -

    Identity Metadata: ``session_id``

    Attributes
    ----------
    data: CoreData
        Indices of the events in the session.
    session_id: str
        Session's identifier.

    Methods
    -------

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
    loader = LoaderNPY
    tpe = TargetType("ndarray_float")

    # --- Key Features -----------------------------------------------------------------------------

    def __init__(
        self,
        session_id: str,
        data: Optional[Union[CoreData, np.ndarray]] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.session_id = session_id
        # Set data and coordinate attributes via the base class constructor
        super().__init__(data=data)

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
        The raw data of one session is the output of a MATLAB script that parses .m files
        (``exptevents.m``). The script generates a CSV file:


        Returns
        -------
        Data

        Raises
        ------
        ValueError
            If the shape of the loaded data is not ``(2, nspikes)``
        """
        # Load numpy array via LoaderNPY
        raw = self.loader(path=self.path, tpe=self.tpe).load()
        print("RAW", raw.shape)
        # Create new instance filled with the loaded data
        obj = self.__class__(self.session_id)  # TODO
        self.__dict__.update(obj.__dict__)
