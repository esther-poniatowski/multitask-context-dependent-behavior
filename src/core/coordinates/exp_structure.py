#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.exp_structure` [module]

Coordinates for labelling the sequential structure of the experiment (positional information).

Classes
-------
`CoordPosition` (Generic)
`CoordRecNum`
`CoordBlock`
`CoordSlot`
"""

from typing import TypeVar, Type, Dict

import numpy as np

from coordinates.base_coord import Coordinate
from core.entities.exp_structure import Recording, Block, Slot


ConcretePosition = TypeVar("ConcretePosition", Recording, Block, Slot)
"""Generic type variable for concrete position entities."""


class CoordPosition(Coordinate[np.int64, ConcretePosition]):
    """
    Coordinate labels for referencing measurements within a whole experiment.

    Each measurement is uniquely identified within the whole experiment by three "positions":

    - Recording number of the session
    - Block within a session
    - Slot within a block

    Class Attributes
    ----------------
    ENTITY : Type[Position]
        Subclass of `Position` corresponding to the type of positional information which is
        represented by the coordinate.
    DTYPE : Type[np.int64]
        Data type of the position labels, always integer.
    SENTINEL : int
        Sentinel value marking missing or unset position labels, usually ``-1`` since positional
        information is always positive.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any], np.int64]
        Position labels for the measurements.
        Shape: ``(n_smpl,)`` with ``n_smpl`` the number of samples.

    Methods
    -------
    `count_by_lab`

    Notes
    -----
    Those coordinates are only used with data sets associated with single units. They are not used
    with data sets associated with pseudo-populations, since the positional information is lost when
    creating pseudo-trials.

    See Also
    --------
    :class:`core.coordinates.base_coord.Coordinate`
    :mod:`core.entities.exp_structure`
    """

    ENTITY: Type[ConcretePosition]
    DTYPE = np.int64
    SENTINEL: int = -1

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{pos}: {n}" for pos, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    def count_by_lab(self) -> Dict[ConcretePosition, int]:
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
        return {self.ENTITY(pos): np.sum(self == pos) for pos in np.unique(self)}


class CoordRecNum(CoordPosition[Recording]):
    """
    Coordinate labels for recording numbers.
    """

    ENTITY = Recording


class CoordBlock(CoordPosition[Block]):
    """
    Coordinate labels for blocks within one session.

    Warning
    -------
    Labels in the coordinate start at 1 (not 0) and extend until the maximum number of trials in the
    session(s) encountered by the unit.
    """

    ENTITY = Block


class CoordSlot(CoordPosition[Slot]):
    """
    Coordinate labels for slots within one block.

    See Also
    --------
    :class:`core.entities.Slot`
    :class:`core.coordinates.CoordPosition`
    """

    ENTITY = Slot
