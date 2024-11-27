#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.raw_spk_times` [module]
"""
from types import MappingProxyType

import numpy as np

from core.constants import SMPL_RATE
from core.coordinates.exp_structure_coord import CoordBlock
from core.data_structures.base_data_struct import DataStructure
from core.data_structures.core_data import Dimensions, CoreData


class SpikeTimesRaw(DataStructure):
    """
    Raw spiking times for one unit (neuron) in one session of the experiment.

    Key Features
    ------------
    Dimensions : ``spikes``

    Coordinates: ``block``

    Identity Metadata: ``unit_id``, ``session_id``

    Descriptive Metadata: ``smpl_rate``

    Attributes
    ----------
    data : CoreData
        Spiking times for the unit in the session (in seconds).
        Times are reset at 0 in each block.
    block : CoordBlock
        Coordinate for the block of trials in which each spike occurred within the session.
        Ascending order, from 1 to the number of blocks in the session, with contiguous duplicates.
        Example: ``1111122223333...``
    unit_id : str
        Unit's identifier.
    session_id : str
        Session's identifier.
    smpl_rate : float, default=`core.constants.SMPL_RATE`
        Sampling time for the recording (in seconds).
    n_blocks : int
        (Property) Number of blocks in the session.

    Methods
    -------
    `get_block`

    Notes
    -----
    This data structure represents the entry point of the data analysis. Therefore, it tightly
    reflects to the raw data, without additional pre-processing. It is not meant to be saved,
    therefore no saver is defined.

    Warning
    -------
    Each spiking time (in seconds) is *relative* to the beginning of the *block* of trials in which
    is occurred (i.e. time origin is reset at 0 in each block).

    See Also
    --------
    `core.coordinates.exp_structure.CoordBlock`
    """

    # --- Schema Attributes ---
    dims = Dimensions("spikes")
    coords = MappingProxyType({"block": CoordBlock})
    coords_to_dims = MappingProxyType({"block": Dimensions("spikes")})
    identifiers = ("unit_id", "session_id")

    # --- Key Features -----------------------------------------------------------------------------

    def __init__(
        self,
        unit_id: str,
        session_id: str,
        smpl_rate: float = SMPL_RATE,
        data: CoreData | None = None,
        block: CoordBlock | None = None,
    ) -> None:
        # Set sub-class specific metadata
        self.unit_id = unit_id
        self.session_id = session_id
        self.smpl_rate = smpl_rate
        # Set data and coordinate attributes via the base class constructor
        coords = {"block": block} if block is not None else {}
        super().__init__(data=data, **coords)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}>: Unit {self.unit_id}, Session {self.session_id}\n"
            + super().__repr__()
        )

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the session."""
        if len(self.block) != 0:  # avoid ValueError in max() if empty array (no spikes)
            return self.block.max()
        else:
            return 0

    def get_block(self, block: int):
        """
        Extract the spiking times which occurred in one block of trials.

        Parameters
        ----------
        block : int
            Block number, comprised between 1 and the total number of blocks.

        Returns
        -------
        raw_data : CoreData
            Spiking times for the unit in the session which occurred in the specified block.

        Warning
        -------
        This method returns the underlying data as a numpy array instead of a new instance of the
        class. It is aimed to extract and manipulate data efficiently in builder pipelines.
        """
        return self.data[self.block == block]

    def format(self, raw: np.ndarray) -> None:
        """
        Retrieve data from a file and extract separately the spiking times and the blocks of trials.

        Arguments
        ---------
        raw : np.ndarray
            Raw data for one unit in one session, loaded from a file.
            Data type: ``float``. Shape: ``(2, nspikes)``.

            Structure :

            - ``raw[0]`` : Block of trials in which each spike occurred.
            - ``raw[1]`` : Spiking times.


        Notes
        -----
        Conversions:

        - The first raw is converted to `CoordBlock`, after being casted to integers. It is
          necessary since the the raw data recovered from the files is a numpy array of type
          ``float``.
        - The second raw is converted to `CoreData` since it contains the spiking times (actual
          values to analyze).

        Raises
        ------
        ValueError
            If the shape of the raw data is not ``(2, nspikes)``
        """
        # Check shape
        if raw.ndim != 2 or raw.shape[0] != 2:
            raise ValueError(f"Invalid shape: {raw.shape}. Expected: (2, nspikes).")
        # Extract data
        block = CoordBlock(values=raw[0].astype(int))  # convert to integer
        data = CoreData(raw[1])
        # Filled with new data (base class methods)
        self.set_data(data)
        self.set_coord("block", block)
