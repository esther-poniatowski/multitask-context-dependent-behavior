#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.raw_spk_times` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Self, Union

import numpy as np

from core.constants import SMPL_RATE
from core.coordinates.exp_structure import CoordBlock
from core.data_structures.base_data_struct import DataStructure
from core.data_structures.core_data import Dimensions, CoreData
from utils.io_data.formats import TargetType
from utils.io_data.loaders.impl_loaders import LoaderNPY
from utils.storage_rulers.impl_path_rulers import SpikeTimesRawPath


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
    data: CoreData
        Spiking times for the unit in the session (in seconds).
        Times are reset at 0 in each block.
    block: CoordBlock
        Coordinate for the block of trials in which each spike occurred within the session.
        Ascending order, from 1 to the number of blocks in the session, with contiguous duplicates.
        Example: ``1111122223333...``
    unit_id: str
        Unit's identifier.
    session_id: str
        Session's identifier.
    smpl_rate: float, default=:obj:`core.constants.SMPL_RATE`
        Sampling time for the recording (in seconds).
    n_blocks: int
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

    # --- IO Handlers ---
    path_ruler = SpikeTimesRawPath
    loader = LoaderNPY
    tpe = TargetType("ndarray_float")

    # --- Key Features -----------------------------------------------------------------------------

    def __init__(
        self,
        unit_id: str,
        session_id: str,
        smpl_rate: float = SMPL_RATE,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        block: Optional[Union[CoordBlock, np.ndarray]] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.unit_id = unit_id
        self.session_id = session_id
        self.smpl_rate = smpl_rate
        # Set data and coordinate attributes via the base class constructor
        super().__init__(data=data, block=block)

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

    def get_block(self, block: int) -> Self:
        """
        Extract the spiking times which occurred in one block of trials.

        Parameters
        ----------
        block: int
            Block number, comprised between 1 and the total number of blocks.

        Returns
        -------
        SpikeTimesRaw
            Spiking times for the unit in the specified block.
        """
        mask = self.block == block
        new_data = self.data[mask]
        new_block = self.block[mask]
        sub_obj = self.__class__(
            unit_id=self.unit_id,
            session_id=self.session_id,
            smpl_rate=self.smpl_rate,
            data=new_data,
            block=new_block,
        )
        return sub_obj

    # --- IO Handling ------------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.unit_id, self.session_id)

    def load(self) -> None:
        """
        Retrieve data from a file and extract separately the spiking times and the blocks of trials.

        Notes
        -----
        The raw data of one unit in one session is stored in a numpy array.
        Shape: ``(2, nspikes)``.
        ``raw[0]`` : Block of trials in which each spike occurred.
        ``raw[1]`` : Spiking times.
        Data type: ``float``.
        Thus, both spiking times and trial blocks are stored under type ``float``.
        Here, blocks of trials are converted to integers to match the format expected by the
        coordinate `CoordBlock`.

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
        # Check shape
        if raw.ndim != 2 or raw.shape[0] != 2:
            raise ValueError(f"Invalid shape: {raw.shape}. Expected (2, nspikes).")
        # Extract data
        block = CoordBlock(values=raw[0].astype(np.int64))  # convert to integer
        t_spk = raw[1]
        # Create new instance filled with the loaded data
        obj = self.__class__(self.unit_id, self.session_id, self.smpl_rate, data=t_spk, block=block)
        self.__dict__.update(obj.__dict__)
