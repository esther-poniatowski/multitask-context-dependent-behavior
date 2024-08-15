#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.raw_spk_times` [module]
"""
from pathlib import Path
from types import MappingProxyType
from typing import Optional, cast, Self

import numpy as np
import numpy.typing as npt

from core.constants import SMPL_RATE
from core.coordinates.exp_structure import CoordBlock
from core.data_structures.base import Data
from utils.io_data.formats import TargetType
from utils.io_data.loaders.impl import LoaderNPY
from utils.path_system.storage_rulers.impl import RawSpkTimesPath


class RawSpkTimes(Data):
    """
    Raw spiking times for one unit (neuron) in one session of the experiment.

    Key Features
    ------------
    Data       : ``data`` (type ``npt.NDArray[np.float64]``)
    Dimensions : ``time``
    Coordinates: ``block`` (type ``CoordBlock``)
    Metadata   : ``unit_id``, ``session_id``, ``smpl_rate``

    Attributes
    ----------
    data: npt.NDArray[np.float64]
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
    smpl_rate: float
        Sampling time for the recording (in seconds).
        Default: :obj:`core.constants.SMPL_RATE`

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
    :class:`core.coordinates.exp_structure.CoordBlock`
    """

    dim2coord = MappingProxyType({"time": frozenset(["block"])})
    coord2type = MappingProxyType({"block": CoordBlock})
    path_ruler = RawSpkTimesPath
    loader = LoaderNPY
    tpe = TargetType("ndarray_float")

    def __init__(
        self,
        unit_id: str,
        session_id: str,
        smpl_rate: float = SMPL_RATE,
        data: Optional[npt.NDArray[np.float64]] = None,
        block: Optional[CoordBlock] = None,
    ) -> None:
        # Set sub-class specific metadata
        self.unit_id = unit_id
        self.session_id = session_id
        self.smpl_rate = smpl_rate
        # Declare data and coordinate attributes (avoid type errors)
        self.data: npt.NDArray[np.float64]
        self.block: CoordBlock
        # Set data and coordinate attributes
        super().__init__(data=data, block=block)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}>: Unit {self.unit_id}, Session {self.session_id}\n"
            + super().__repr__()
        )

    @property
    def path(self) -> Path:
        return self.path_ruler().get_path(self.unit_id, self.session_id)

    def load(self) -> Self:  # pylint: disable=undefined-variable
        """
        Retrieve data from a file and extract separately the spiking times and the blocks of trials.

        Notes
        -----
        The raw data of one neuron in one session is stored in a numpy array.
        Shape: ``(2, nspikes)``.
        ``raw[0]`` : Block of trials in which each spike occurred.
        ``raw[1]`` : Spiking times.
        Data type: ``float``.
        Thus, both spiking times and trial blocks are stored under type ``float``.
        Here, blocks of trials are converted to integers to match the format expected by the
        coordinate :class:`CoordBlock`.

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
        # Check shape
        if raw.ndim != 2 or raw.shape[0] != 2:
            raise ValueError(f"Invalid shape: {raw.shape}. Expected (2, nspikes).")
        # Extract data
        block = CoordBlock(values=raw[0].astype(np.int64))  # convert to integer
        t_spk = raw[1]
        # Create new instance filled with the loaded data
        obj = self.__class__(self.unit_id, self.session_id, self.smpl_rate, data=t_spk, block=block)
        return obj

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the session."""
        if len(self.block) != 0:  # avoid ValueError in max() if empty array (no spikes)
            return self.block.max()
        else:
            return 0

    def get_block(self, blk: int) -> Self:
        """
        Extract the spiking times which occurred in one block of trials.

        Parameters
        ----------
        blk: int
            Block number, comprised between 1 and the total number of blocks.

        Returns
        -------
        RawSpkTimes
            Spiking times for the unit in the specified block.
        """
        mask = cast(npt.NDArray[np.bool_], self.block == blk)  # see overloaded __getitem__
        data = self.data[mask]
        block = self.block[mask]
        sub_obj = self.__class__(self.unit_id, self.session_id, self.smpl_rate, data, block)
        return sub_obj
