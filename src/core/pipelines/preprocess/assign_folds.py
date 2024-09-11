#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.pipelines.preprocess.assign_folds` [module]

Classes
-------
:class:`FoldsAssigner`

Notes
-----
Folds correspond to subsets of samples (trials) used to train and test a model in cross-validation.

In this module, the class :class:`FoldsAssigner` is only responsible of assigning each sample to one
fold. The splitting into training and testing sets is carried out in the :class:`CoordFold` class
itself by selecting the samples based on their fold assignment (see methods :meth:`get_train` and
:meth:`get_test`). This allows direct access to the samples in each set without resorting to an
external cross-validation tool or class.

The assignment of trials to folds should be performed *by unit*, before constructing pseudo-trials
via hierarchical bootstrap. This ensures that trials are combined within each folds and prevents
data leakage across folds.

The assignment of trials should also be *stratified* by condition (task, context, stimulus, error),
to balance the distribution of the distinct types of trials across folds.

Implementation
--------------
Procedure for fold assignment:

1. Keep valid trial (if necessary).
2. Split samples by condition (task, context, stimulus, error).
2. For each condition, shuffle the trials and distribute them in k groups.
3. Assign each trial to a fold by selecting the corresponding group.

Shuffling before splitting aims to balance across folds the task variables which have not been
considered in the stratification process, namely the positional information (recording number, block
number, slot number). This precaution avoids to include in the models any misleading temporal drift
of the neuronal activity.

"""
from typing import Optional

import numpy as np
import numpy.typing as npt


class FoldsAssigner:
    """
    Assign samples (trials) to folds for cross-validation.

    Attributes
    ----------
    k: int
        Number of folds in which the samples will be divided.
    n_samples : int
        Number of samples to assign to folds.
    strata : npt.NDArray[np.int64], default=None
        Labels of strata to stratify the samples. Shape: ``(n_samples,)``.
        If None, all samples are treated as belonging to a single stratum.
    seed : int, default=0
        Random state for reproducibility in shuffling samples before fold assignment.
    _folds : npt.NDArray[np.int64]
        Fold assignment for each sample. Shape: ``(n_samples,)``.
        It is computed lazily on first access and cached for subsequent accesses.
    folds : npt.NDArray[np.int64]
        Access to the fold assignment via the property.

    Methods
    -------
    :meth:`set_seed`
    :meth:`assign`
    """

    def __init__(
        self, k: int, n_samples: int, strata: Optional[npt.NDArray[np.int64]] = None, seed: int = 0
    ):
        self.n_samples = n_samples
        self._validate_k(k)
        self.k = k
        self.seed = seed
        if strata is None:
            strata = self.create_strata()
        else:
            self._validate_strata(strata)
        self.strata = strata
        self._folds: Optional[npt.NDArray[np.int64]] = None

    @property
    def folds(self) -> npt.NDArray[np.int64]:
        """Fold assignment for each sample."""
        if self._folds is None:
            self._folds = self.assign()
        return self._folds

    def _validate_k(self, k: int) -> None:
        """
        Validate the number of folds.

        Raises
        ------
        ValueError
            If the number of folds is larger than the number of samples.
        """
        if k > self.n_samples:
            raise ValueError(
                f"[ERROR] Number of folds: {k}, expected <= number of samples: {self.n_samples}"
            )

    def _validate_strata(self, strata: npt.NDArray[np.int64]) -> None:
        """
        Validate the strata used for stratification if provided.

        Raises
        ------
        ValueError
            If the strata are not provided or not of the correct shape.
        """
        if strata.shape != (self.n_samples,):
            raise ValueError(
                f"[ERROR] Length of `strata`: {strata.shape}, expected `n_samples`: {(self.n_samples,)}"
            )

    def create_strata(self) -> npt.NDArray[np.int64]:
        """
        Create default strata labels, treating all samples as belonging to a single stratum.

        Returns
        -------
        strata : npt.NDArray[np.int64]
            Array of zeros with the same length as the number of samples.
        """
        return np.zeros(self.n_samples, dtype=np.int64)

    def set_seed(self) -> None:
        """
        Set the random seed for reproducibility.

        See Also
        --------
        :func:`np.random.seed`: Set the random seed for reproducibility.
        """
        np.random.seed(self.seed)

    def assign(self) -> npt.NDArray[np.int64]:
        """
        Assign folds to trials, stratified by condition.

        Returns
        -------
        folds : npt.NDArray[np.int64]
            See :attr:`folds`.

        See Also
        --------
        :func:`np.random.shuffle`
        :func:`np.where`
            Get the indices where a condition is matched.
            Output (here): ``(array([i1, i2, ..., iN]),)`` where ``N`` is the number of
            indices in the considered stratum. Only the first element of the tuple is used.
        :func:`np.array_split(arr, n)`
            Split an array of length ``l`` into ``n`` sub-arrays of maximally equal size.
            Output (list): ``l % n`` sub-arrays of size ``l // n + 1`` and ``n - l % n`` sub-arrays
            of size ``l // n``.
        """
        self.set_seed()
        folds = np.zeros(self.n_samples, dtype=np.int64)  # initialize folds
        for stratum in np.unique(self.strata):
            idx_stratum = np.where(self.strata == stratum)[0]  # indices of samples in the stratum
            np.random.shuffle(idx_stratum)  # shuffle samples before splitting
            split_indices = np.array_split(idx_stratum, self.k)  # split samples into k groups
            for i_fold, idx_samples in enumerate(split_indices):
                folds[idx_samples] = i_fold
        return folds
