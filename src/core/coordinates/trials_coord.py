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


class CoordFolds(Coordinate[np.int64]):
    """
    Coordinate labels indicating the fold to which each trial belongs.

    Class Attributes
    ----------------
    DTYPE : np.int64
        Data type of the fold labels, always integer.
    SENTINEL : int
        Sentinel value marking missing or unset index, here ``-1``.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any, ...], np.int64]
        Fold labels for each trial.
        Shape: Variable depending on the number of dimensions (for multiple units in a population,
        multiple ensembles, multiple conditions, etc.).
        For a 1D coordinate, shape is ``(n_smpl,)`` (only a "trials" dimension).

    Notes
    -----
    No specific entity is associated with fold labels.

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    """

    # No ENTITY
    DTYPE = np.float64
    SENTINEL = -1


class CoordPseudoTrialsIdx(Coordinate[np.int64]):
    """
    Coordinate labels for trial indices to pick from the raw data to form pseudo-trials.

    One dimension is associated to the final pseudo-trials.

    Class Attributes
    ----------------
    DTYPE : np.int64
        Data type of the trial indices, always integer.
    SENTINEL : int
        Sentinel value marking missing or unset index, here ``-1``.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any, ...], np.int64]
        Trial indices to select from the raw data to form pseudo-trials.
        Shape: Variable, depending on the number of dimensions (for multiple units in a population,
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
