#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates.trials` [module]

Coordinate for labelling trials : errors, folds.

Classes
-------
:class:`CoordError`
:class:`CoordFold`
"""

from typing import Optional, Dict, List

import numpy as np
import numpy.typing as npt

from core.coordinates.base import Coordinate


class CoordError(Coordinate):
    """
    Coordinate labels for error and success trials in the data set.

    Attributes
    ----------
    values: npt.NDArray[np.bool_]
        Booleans indicating for each measurement whether it
        occurred during an error trial (True) or correct trial (False).

    Methods
    -------
    :meth:`count_by_lab`

    See Also
    --------
    :class:`mtcdb.coordinates.base.Coordinate`
    """

    def __init__(self, values: npt.NDArray[np.bool_]):
        super().__init__(values=values)

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = f"Correct: {counts[False]}, Error: {counts[True]}"
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    @staticmethod
    def build_labels(n_smpl: int) -> npt.NDArray[np.str_]:
        """
        Build basic labels filled with correct trials.

        Parameters
        ----------
        n_smpl: int
            Number of samples, i.e. of labels.

        Returns
        -------
        values: npt.NDArray[np.bool_]
            Labels filled with ``False``.
        """
        return np.full(n_smpl, False, dtype=np.bool_)

    def count_by_lab(self) -> Dict[bool, int]:
        """
        Count the number of samples for correct and error trials respectively.

        Returns
        -------
        n_smpl: Dict[bool, int]
            Number of samples in error and correct trials.
        """
        return {True: np.sum(self.values).astype(int), False: np.sum(~self.values).astype(int)}


class CoordFold(Coordinate):
    """
    Coordinate labels for the samples' folds in cross-validation.

    Methods
    -------
    :meth:`count_by_lab`
    :meth:`get_test`
    :meth:`get_train`

    Attributes
    ----------
    values: npt.NDArray[np.int64]
        Fold identifiers, starting from 0.
    k: int
        Number of folds.

    Notes
    -----
    The number of folds is computed from the unique values in the coordinate.

    The test set for a fold contains the samples labelled *with* the fold, whereas the training set
    contains the samples *not* labelled with the fold. This convention for cross-validation ensures
    that the training and testing sets of samples are disjoint. The test set withholds a minor
    subset of samples while the training set withholds the broad subset.
    """

    def __init__(self, values: npt.NDArray[np.int64], k: Optional[int] = None):
        super().__init__(values=values)
        if k is not None:
            self.k = k
        else:
            self.k = len(np.unique(values))

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = ", ".join([f"Fold {fold}: {n}" for fold, n in enumerate(counts)])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    @staticmethod
    def build_labels(n_smpl: int) -> npt.NDArray[np.int64]:
        """
        Build basic fold labels in which all the samples are gathered in a single fold.

        Parameters
        ----------
        n_smpl: int
            Number of samples.

        Returns
        -------
        values: npt.NDArray[np.int64]
            Labels filled with ``0``.
        """
        return np.zeros(n_smpl, dtype=np.int64)

    def count_by_lab(self) -> List[int]:
        """
        Count the number of samples in each fold.

        Returns
        -------
        n_smpl: List[int]
            Number of samples in each fold.
            Each index corresponds to a fold number.
        """
        return [np.sum(self.values == fold) for fold in range(self.k)]

    def get_test(self, fold: int) -> npt.NDArray[np.bool_]:
        """
        Identify the test samples for one fold (labelled *with* the fold number).

        Parameters
        ----------
        fold: int
            Fold to select for testing.

        Returns
        -------
        mask: npt.NDArray[np.bool_]
            Boolean mask for the samples used for testing.
            Shape : ``(n_smpl,)``
        """
        return self.values == fold

    def get_train(self, fold: int) -> npt.NDArray[np.bool_]:
        """
        Identify the train samples for one fold (labelled *without* the fold number).

        Parameters
        ----------
        fold: int
            Fold to select for training.

        Returns
        -------
        mask: npt.NDArray[np.bool_]
            Boolean mask for the samples used for training.
            Shape : ``(n_smpl,)``
        """
        return self.values != fold
