#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.assign_folds` [module]

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

from typing import Literal, overload, TypeAlias, Any, Tuple, Union, List

import numpy as np

from core.processors.base_processor import Processor

FoldLabels: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for fold labels assigned to each sample."""

FoldMembers: TypeAlias = List[np.ndarray[Tuple[Any], np.dtype[np.int64]]]
"""Type alias for fold members, i.e. samples contained in each fold."""


class FoldAssigner(Processor):
    """
    Assign samples (trials) to folds for cross-validation.

    Configuration Attributes
    ------------------------
    k : int
        Number of folds in which the samples will be divided.

    Processing Arguments
    --------------------
    n_samples : int
        Number of samples to assign to folds.
        .. _n_samples:
    mode : Literal["labels", "members"], default="labels"
        Return either the fold labels or the fold members.

    Returns
    -------
    fold_members: FoldMembers
        Indices of the samples contained in each fold.
        .. _fold_members:

        Number of sub-arrays: ``k``. Shapes:

        - ``n % k`` sub-arrays of size ``n // k + 1`` (to distribute the remainder's elements)
        - ``k - (n % k)`` sub-arrays of size ``n // k``.

    fold_labels : FoldLabels
        Fold labels assigned to each sample. Shape: ``(n_samples,)``.
        .. _fold_labels:

    Methods
    -------
    `assign`
    `labels_to_members`
    `members_to_labels`

    Examples
    --------
    Assign 10 samples to 3 folds:

    - ``n % k = 1`` array of size ``n // k + 1 = 4``.
    - ``k - (n % k) = 2`` arrays of size ``n // k = 3``.

    >>> assigner = FoldAssigner(k=3)
    >>> fold_members = assigner.process(n_samples=10)
    >>> fold_members
    [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
    >>> fold_labels = assigner.process(n_samples=10, mode="labels")
    >>> fold_labels
    array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random = True

    def __init__(self, k: int):
        super().__init__(k=k)

    # --- Processing Methods -----------------------------------------------------------------------

    @overload
    def _process(self, mode: Literal["labels"] = "labels", **input_data: Any) -> FoldLabels: ...

    @overload
    def _process(self, mode: Literal["members"] = "members", **input_data: Any) -> FoldMembers: ...

    def _process(self, mode: str = "labels", **input_data: Any) -> Union[FoldLabels, FoldMembers]:
        """Implement the template method called in the base class `process` method."""
        n_samples = input_data["n_samples"]
        fold_members = self.assign(n_samples)
        if mode == "members":
            return fold_members
        elif mode == "labels":
            return self.members_to_labels(fold_members)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def assign(self, n_samples: int) -> FoldMembers:
        """
        Assign each sample to one fold.

        Arguments
        ---------
        n_samples : int
            See the argument :ref:`n_samples`.

        Returns
        -------
        fold_members: FoldMembers
            See the return value :ref:`fold_members`.

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
        :func:`np.array_split(arr, k)`
            Split an array of length ``n`` into ``k`` sub-arrays of maximally equal size, so that
            the size difference between any two sub-arrays is at most 1.
        """
        idx = np.arange(n_samples)  # indices of the samples
        np.random.shuffle(idx)  # shuffle samples before splitting
        fold_members = np.array_split(idx, self.k)  # split samples into k groups
        return fold_members

    # --- Conversion Methods -----------------------------------------------------------------------

    @staticmethod
    def labels_to_members(fold_labels: FoldLabels) -> FoldMembers:
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
        fold_members = [np.where(fold_labels == i)[0] for i in range(n_folds)]
        return fold_members

    @staticmethod
    def members_to_labels(fold_members: FoldMembers) -> FoldLabels:
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

    @staticmethod
    def eval_min_count(k: int, n_samples: int) -> int:
        """
        Evaluate the minimum count of samples assigned to each fold based on the total number of
        samples. Useful to determine the number of pseudo-trials fo form.

        Rule: `count_min = n_samples // k` (see the `assign` method).

        Arguments
        ---------
        k : int
            Number of folds.
        n_samples : int
            Number of samples to assign to folds.

        Returns
        -------
        count : int
            Number of samples in each fold after assignment.
        """
        return n_samples // k
