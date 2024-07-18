#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates.exp_structure` [module]

Coordinates for labelling the sequential structure of the experiment (positional information).

Classes
-------
:class:`CoordPosition` (Generic)
:class:`CoordRecNum`
:class:`CoordBlock`
:class:`CoordSlot`
"""

from typing import TypeVar, Type, Generic, Dict

import numpy as np
import numpy.typing as npt

from mtcdb.coordinates.base import Coordinate
from mtcdb.core_objects.exp_structure import Recording, Block, Slot


P = TypeVar("P", Recording, Block, Slot)
"""Generic type variable for positions."""


class CoordPosition(Coordinate, Generic[P]):
    """
    Coordinate labels for referencing measurements within a whole experiment.

    Each measurement is uniquely identified within the whole experiment by three "positions":

    - Recording number of the session
    - Block within a session
    - Slot within a block

    Class Attributes
    ----------------
    position: Type[P]
        Subclass of :class:`Position` corresponding to the type of positional information which is
        represented by the coordinate.
        It has to be defined in each subclass to match the specific position type.

    Attributes
    ----------
    values: npt.NDArray[np.int64]
        Position labels for the measurements. All the positions are represented by integer values,
        matching the values of the corresponding position objects.

    Methods
    -------
    :meth:`count_by_lab`

    Notes
    -----
    Those coordinates are only used with data sets associated with single units. They are not used
    with data sets associated with pseudo-populations, since the positional information is lost when
    creating pseudo-trials.

    See Also
    --------
    :class:`mtcdb.coordinates.base.Coordinate`
    :mod:`mtcdb.core_objects.exp_structure`
    """

    position: Type[P]

    def __init__(self, values: npt.NDArray[np.int64]):
        super().__init__(values=values)

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{pos!r}: {n}" for pos, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    def count_by_lab(self) -> Dict[P, int]:
        """
        Count the number of samples in each position.

        Returns
        -------
        n_smpl: Dict[int, int]
            Number of samples in each position.

        Implementation
        --------------
        Because the options for the positions are not known in advance, the counts are provided for
        all the distinct positions present in the coordinate.
        """
        return {pos: np.sum(self.values == pos) for pos in np.unique(self.values)}

    @staticmethod
    def build_labels(
        n_smpl: int, pos: P
    ) -> npt.NDArray[np.int64]:  # pylint: disable=arguments-differ
        """
        Build basic labels filled with a *single* position.

        Parameters
        ----------
        pos: P
            Position whose value is the single label.
        n_smpl: int
            Number of samples, i.e. of labels.

        Returns
        -------
        values: npt.NDArray[np.int64]
            Position labels filled with a single position.
            Length: ``n_smpl``, Unique value: ``pos``.
        """
        return np.full(n_smpl, pos.value, dtype=np.int64)


class CoordRecNum(CoordPosition[Recording]):
    """
    Coordinate labels for recording numbers.

    See Also
    --------
    :class:`mtcdb.core_objects.Recording`
    :class:`mtcdb.coordinates.CoordPosition`
    """

    position = Recording

    def __init__(self, values: npt.NDArray[np.int64]):
        super().__init__(values=values)


class CoordBlock(CoordPosition[Block]):
    """
    Coordinate labels for blocks within one session.

    Warning
    -------
    Labels in the coordinate start at 1 (not 0) and extend until
    the maximum number of trials in the session(s) encountered by the unit.

    See Also
    --------
    :class:`mtcdb.core_objects.Block`
    :class:`mtcdb.coordinates.CoordPosition`
    """

    position = Block

    def __init__(self, values: npt.NDArray[np.int64]):
        super().__init__(values=values)


class CoordSlot(CoordPosition[Slot]):
    """
    Coordinate labels for slots within one block.

    See Also
    --------
    :class:`mtcdb.core_objects.Slot`
    :class:`mtcdb.coordinates.CoordPosition`
    """

    position = Slot

    def __init__(self, values: npt.NDArray[np.int64]):
        super().__init__(values=values)
