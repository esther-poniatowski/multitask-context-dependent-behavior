#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.trials_coord` [module]

Coordinates for labelling trials in a data set.

Classes
-------
CoordFolds
CoordPseudoTrialsIdx
"""

import numpy as np

from core.coordinates.base_coord import Coordinate
from core.attributes.trial_analysis_labels import Fold, TrialIndex


class CoordFolds(Coordinate[np.int64, Fold]):
    """
    Coordinate labels indicating the fold to which each trial belongs.

    Class Attributes
    ----------------
    DTYPE : np.int64
        Data type of the fold labels, always integer.
    SENTINEL : int
        Sentinel value marking missing or unset index, here ``-1``.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.int64]
        Fold labels for each trial.
        Shape: Variable depending on the number of dimensions (for multiple units in a population,
        multiple ensembles, multiple conditions, etc.).
        For a 1D coordinate: ``(n_smpl,)`` (only a "trials" dimension).

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    """

    ATTRIBUTE = Fold
    DTYPE = np.int64
    SENTINEL = -1


class CoordPseudoTrialsIdx(Coordinate[np.int64, TrialIndex]):
    """
    Coordinate labels for trial indices to pick from the raw data to form pseudo-trials.

    One dimension is associated to the final pseudo-trials.

    Class Attributes
    ----------------
    DTYPE : np.int64
        Data type of the trial indices, always integer.
    SENTINEL : int
        Sentinel value marking missing or unset index, here ``-1``.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.int64]
        Trial indices to select from the raw data to form pseudo-trials.
        Shape: Variable, depending on the number of dimensions (for multiple units in a population,
        multiple ensembles, multiple folds, multiple conditions, etc.).
        For a 1D coordinate, shape is ``(n_smpl,)`` (only a "trials" dimension).

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    """

    ATTRIBUTE = TrialIndex
    DTYPE = np.int64
    SENTINEL = -1
