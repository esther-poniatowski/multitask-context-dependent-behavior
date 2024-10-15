#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.trials` [module]

Coordinate for labelling trials : errors, folds.

Classes
-------
`CoordError`
"""

from typing import Dict, Self

import numpy as np

from coordinates.base_coord import Coordinate


class CoordError(Coordinate[np.bool_]):
    """
    Coordinate labels for error and success trials in the data set.

    Class Attributes
    ----------------
    DTYPE : np.bool_
        Data type of the error labels, always boolean.
    SENTINEL : bool
        Sentinel value marking missing or unset error labels, ``False`` by default.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any], np.dtype[np.bool_]]
        Booleans indicating for each measurement whether it occurred during an error trial (True) or
        correct trial (False). One dimensional.

    Methods
    -------
    `count_by_lab`

    Notes
    -----
    No entity is specified for behavioral outcomes.

    See Also
    --------
    :class:`core.coordinates.base_coord.Coordinate`
    """

    DTYPE = np.bool_
    SENTINEL = False

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = f"Correct: {counts[False]}, Error: {counts[True]}"
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    def count_by_lab(self) -> Dict[bool, int]:
        """
        Count the number of samples for correct and error trials respectively.

        Returns
        -------
        n_smpl : Dict[bool, int]
            Number of samples in error and correct trials.
        """
        return {True: np.sum(self).astype(int), False: np.sum(not self).astype(int)}

    @classmethod
    def build_labels(cls, n_smpl: int, value: bool = False) -> Self:
        """
        Build basic labels filled with one single behavioral outcome.

        Parameters
        ----------
        n_smpl : int
            Number of samples, i.e. of labels.

        Returns
        -------
        values : Self
            Labels filled with the specified value.
        """
        return cls(np.full(n_smpl, value))
