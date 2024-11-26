#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.exp_structure` [module]

Coordinates for labelling the sequential structure of the experiment (positional information).

Classes
-------
CoordExpStructure (Generic)
CoordRecNum
CoordBlock
CoordSlot
"""

from typing import TypeVar, Type, Dict

import numpy as np

from coordinates.base_coord import Coordinate
from core.attributes.exp_structure import Recording, Block, Slot


AnyExpStructure = TypeVar("AnyExpStructure", Recording, Block, Slot)
"""Generic type variable for concrete position attributes. Used to keep a generic type for the
experimental factor while specifying the data type of the coordinate labels."""


class CoordExpStructure(Coordinate[np.int64, AnyExpStructure]):
    """
    Coordinate labels for referencing measurements within a whole experiment.

    Each measurement is uniquely identified within the whole experiment by three "positions":

    - Recording number of the session
    - Block within a session
    - Slot within a block

    Class Attributes
    ----------------
    ATTRIBUTE : Type[ExpStructure]
        Subclass of `ExpStructure` corresponding to the type of positional information which is
        represented by the coordinate.
    DTYPE : Type[np.int64]
        Data type of the position labels, always integer.
    SENTINEL : int
        Sentinel value marking missing or unset position labels, usually ``-1`` since positional
        information is always positive.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any], np.int64]
        ExpStructure labels for the measurements.
        Shape: ``(n_smpl,)`` with ``n_smpl`` the number of samples.

    Methods
    -------
    count_by_lab

    Notes
    -----
    Those coordinates are only used with data sets associated with single units. They are not used
    with data sets associated with pseudo-populations, since the positional information is lost when
    creating pseudo-trials.

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    `core.attributes.exp_structure`
    """

    ATTRIBUTE: Type[AnyExpStructure]
    DTYPE = np.int64
    SENTINEL: int = -1

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{pos}: {n}" for pos, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    def count_by_lab(self) -> Dict[AnyExpStructure, int]:
        """
        Count the number of samples in each position.

        Returns
        -------
        n_smpl : Dict[int, int]
            Number of samples in each position.

        Implementation
        --------------
        Because the options for the positions are not known in advance, the counts are provided for
        all the distinct positions present in the coordinate.
        """
        return {self.ATTRIBUTE(pos): np.sum(self == pos) for pos in np.unique(self)}


class CoordRecNum(CoordExpStructure[Recording]):
    """
    Coordinate labels for recording numbers.
    """

    ATTRIBUTE = Recording


class CoordBlock(CoordExpStructure[Block]):
    """
    Coordinate labels for blocks within one session.

    Warning
    -------
    Labels in the coordinate start at 1 (not 0) and extend until the maximum number of trials in the
    session(s) encountered by the unit.
    """

    ATTRIBUTE = Block


class CoordSlot(CoordExpStructure[Slot]):
    """
    Coordinate labels for slots within one block.
    """

    ATTRIBUTE = Slot
