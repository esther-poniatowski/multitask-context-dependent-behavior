#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.trials_coord` [module]

Coordinate for labelling time stamps at which measurements were performed.

Classes
-------
`CoordPseudoTrialsIdx`
"""

import numpy as np

from coordinates.base_coord import Coordinate


class CoordPseudoTrialsIdx(Coordinate[np.int64]):
    """
    Coordinate labels for time stamps at which measurements were performed.

    Class Attributes
    ----------------
    DTYPE : np.int64
        Data type of the trial indices, always integer.
    SENTINEL : int
        Sentinel value marking missing or unset index, here ``-1``.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any, ...], np.int64]
        Trial indices.
        Shape : Variable depending on the number of dimensions (for multiple units in a population,
        multiple ensembles, multiple folds, multiple conditions, etc.).
        For a 1D coordinate, shape is ``(n_smpl,)`` (only a "trials" dimension).

    Notes
    -----
    No specific entity is associated with trial indices.

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    """

    # No ENTITY
    DTYPE = np.float64
    SENTINEL = -1
