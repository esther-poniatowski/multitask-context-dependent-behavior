#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.assign_folds` [module]

Classes
-------
`FoldAssigner`

Notes
-----
Folds correspond to subsets of samples (trials) used to train and test a model in cross-validation.

The class `FoldAssigner` is only responsible of assigning each sample to one fold. Actual splitting
samples into training and testing sets is carried out in the `CoordFold` class itself based on these
fold assignments (methods `CoordFold.get_train` and `CoordFold.get_test`). This allows direct access
to the samples in each set through the coordinate, without resorting to external cross-validation
tools.
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member

from typing import TypeAlias, Any, Tuple, Dict

import numpy as np

from core.processors.base_processor import Processor
from core.processors.preprocess.stratify import Strata


FoldLabels: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for fold labels assigned to each sample."""

FoldMembers: TypeAlias = np.ndarray[Tuple[Any, Any], np.dtype[np.int64]]
"""Type alias for fold members, i.e. samples contained in each fold."""


class FoldAssigner(Processor):
    """
    Assign samples (trials) to folds for cross-validation.

    Conventions for the documentation:

    - Attributes: Configuration parameters of the processor, passed to the *constructor*.
    - Arguments: Input data to process, passed to the `process` method (base class).
    - Returns: Output data after processing, returned by the `process` method (base class).

    Attributes
    ----------
    k : int
        Number of folds in which the samples will be divided.

    Arguments
    ---------
    n_samples : int
        Number of samples to assign to folds.
        If not provided, it is inferred from the length of `strata`.
        .. _n_samples:
    strata : Strata
        Labels of strata for stratified assignment. Shape: ``(n_samples,)``.
        If not provided, `n_samples` samples are treated as belonging to a single stratum.
        .. _strata:
    mode : Literal["labels", "members"], default="labels"
        Return either the fold labels or the fold members.

    Returns
    -------
    fold_labels : FoldLabels
        Fold labels assigned to each sample. Shape: ``(n_samples,)``.
        .. _fold_labels:
    fold_members: FoldMembers
        Samples contained in each fold. Shape: ``(k, n_samples // k)``.
        .. _fold_members:

    Methods
    -------
    `assign`
    `labels_to_members`
    `members_to_labels`

    Warning
    -------
    Provide either `n_samples` or `strata` as input to the processor.

    Examples
    --------
    Assign 10 samples to 3 folds:

    >>> assigner = FoldAssigner(k=3)
    >>> fold_labels = assigner.process(n_samples=10)
    >>> print(fold_labels)
    [0 0 1 1 2 2 0 1 2 0]

    Use stratified fold assignment:

    >>> strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    >>> fold_labels = assigner.process(strata=strata)
    >>> print(fold_labels)
    [0 0 1 1 2 2]

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random = True

    def __init__(self, k: int):
        super().__init__(k=k)

    def _pre_process(self, **input_data: Any) -> Dict[str, Any]:
        """
        Ensure consistency between both inputs:

        - Check that exactly one input is provided as argument (not both, nor none).
        - Set the default value for the missing attribute based on the other one.

        Raises
        ------
        ValueError
            If both `n_samples` and `strata` are missing.
            If both `n_samples` and `strata` are provided.
            If the number of samples is lower than the number of folds to form.

        Notes
        -----
        Rules to assign default values:

        - If `strata` is provided, then `n_samples` is equal to the length of `strata`.
        - If `n_samples` is provided, then all samples are treated as belonging to a single stratum,
          therefore `strata` is a zero array of length `n_samples` (single label 0).
        """
        n_samples = input_data.get("n_samples", None)
        strata = input_data.get("strata", None)
        # Set the default value for the missing input
        if strata is None and n_samples is not None:
            strata = np.zeros(n_samples, dtype=np.int64)
        elif n_samples is None and strata is not None:
            n_samples = strata.size
        else:
            raise ValueError("Invalid arguments: provide either `n_samples` or `strata`.")
        # Check if the resulting number of samples is lower than the number of folds
        if n_samples < self.k:
            raise ValueError(f"n_samples: {n_samples} < k: {self.k}")
        # Update input data with new values
        input_data.update({"n_samples": n_samples, "strata": strata})
        return input_data

    def _process(self, mode="labels", **input_data: Any) -> FoldLabels:
        """Implement the template method called in the base class `process` method."""
        strata = input_data["strata"]
        fold_labels = self.assign(strata)
        if mode == "labels":
            return fold_labels
        elif mode == "members":
            return self.labels_to_members(fold_labels)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def assign(self, strata: Strata) -> FoldLabels:
        """
        Assign folds to each sample based on strata labels.

        Arguments
        ---------
        strata: Strata
            See the argument :ref:`strata`.

        Returns
        -------
        fold_labels: FoldLabels
            See the return value :ref:`fold_labels`.

        Implementation
        --------------
        1. Extract the samples by strata.
        2. Within each strata, shuffle the samples to balance across folds the remaining variables
           have not been considered in stratification.
        3. Within the considered strata, distribute the shuffled samples into k groups.
        4. For each sample, associate the fold index based on the group to which it belongs.

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
        fold_labels = np.zeros(strata.size, dtype=np.int64)  # initialize folds
        for stratum in np.unique(strata):
            idx_stratum = np.where(strata == stratum)[0]  # indices of samples in the stratum
            np.random.shuffle(idx_stratum)  # shuffle samples before splitting
            split_indices = np.array_split(idx_stratum, self.k)  # split samples into k groups
            for i_fold, idx_samples in enumerate(split_indices):
                fold_labels[idx_samples] = i_fold
        return fold_labels

    @staticmethod
    def labels_to_members(fold_labels: np.ndarray) -> np.ndarray:
        """
        Convert fold labels to fold members.

        Arguments
        ---------
        fold_labels : np.ndarray
            See the return value :ref:`fold_labels`.

        Returns
        -------
        fold_members : np.ndarray
            See the return value :ref:`fold_members`.
        """
        n_folds = np.max(fold_labels) + 1
        fold_members = np.array([np.where(fold_labels == i)[0] for i in range(n_folds)])
        return fold_members

    @staticmethod
    def members_to_labels(fold_members: np.ndarray) -> np.ndarray:
        """
        Convert fold members to fold labels.

        Arguments
        ---------
        fold_members : np.ndarray
            See the return value :ref:`fold_members`.

        Returns
        -------
        fold_labels : np.ndarray
            See the return value :ref:`fold_labels`.
        """
        n_samples = np.max(fold_members) + 1
        fold_labels = np.full(n_samples, -1, dtype=np.int64)
        for i_fold, idx_samples in enumerate(fold_members):
            fold_labels[idx_samples] = i_fold
        return fold_labels