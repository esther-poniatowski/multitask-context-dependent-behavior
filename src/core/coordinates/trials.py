#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.coordinates.trials` [module]

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
    :class:`core.coordinates.base.Coordinate`
    """

    def __init__(self, values: npt.NDArray[np.bool_]):
        super().__init__(values=values)

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = f"Correct: {counts[False]}, Error: {counts[True]}"
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    # pylint: disable=arguments-differ
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

    # pylint: enable=arguments-differ

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
    Coordinate labels for the fold assignment of each sample in cross-validation.

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
        Number of folds. If not provided at initialization, it is inferred from the maximal value in
        ``values``.

    Notes
    -----
    Cross-validation with a leave-one-out strategy
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    For each fold, the samples are divided into disjoint test and training sets. To withhold a minor
    subset of samples for testing and the broadest subset for training, the samples are distributed
    as follows:

    - Test set: Samples assigned to the considered fold.
    - Training set: Samples assigned to any other fold.

    Representation of fold assignments
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Fold assignments are stored as a *coordinate* rather than a list of numpy arrays containing the
    indices of the samples in each fold. Thereby, the fold assignments are inherently tied to their
    associated data structure. This allows to extract subsets of data while preserving the correct
    fold assignments, since the corresponding subset of the CoordFold coordinate is automatically
    extracted in parallel. In contrast, an approach using using a list of numpy arrays for each fold
    would require re-indexing after any data extraction.

    Warning
    -------
    The output arrays of :meth:`get_test` and :meth:`get_train` are *boolean* masks. If they
    contained ``1`` and ``0`` values instead, they would be interpreted as the *indices* of the
    trials to select, which would lead to pick only trials 0 and 1 multiple times.

    Examples
    --------
    Assume that the data structure `ds` has a dimension `trials` in the first axis and stores a
    CoordFold object in the attribute ``folds``. To select the test samples for fold 2:

    >>> mask_test = ds.folds.get_test(fold=2)
    >>> ds_test_2 = ds[mask_test, ...] # index by the boolean mask along the first axis

    """

    def __init__(self, values: npt.NDArray[np.int64], k: Optional[int] = None):
        super().__init__(values=values)
        if k is not None:
            self.k = k
        else:
            self.k = int(np.max(values)) + 1  # indices start from 0

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = ", ".join([f"Fold {fold}: {n}" for fold, n in enumerate(counts)])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    # pylint: disable=arguments-differ
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

    # pylint: enable=arguments-differ

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
