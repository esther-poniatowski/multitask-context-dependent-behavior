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

from typing import Literal, overload, TypeAlias, Any, Tuple, Union, List

import numpy as np

from core.processors.base_processor import Processor, set_random_state


FoldLabels: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for fold labels assigned to each sample."""

FoldMembers: TypeAlias = List[np.ndarray[Tuple[Any], np.dtype[np.int64]]]
"""Type alias for fold members, i.e. samples contained in each fold."""


class FoldAssigner(Processor):
    """
    Assign samples (trials) to folds for cross-validation.

    Attributes
    ----------
    n_folds : int
        Number of folds in which the samples will be divided.

    Methods
    -------
    `assign`
    `labels_to_members`
    `members_to_labels`

    Examples
    --------
    Assign ``n=10`` samples to ``k=3`` folds:

    - ``n % k = 1`` array of size ``n // k + 1 = 4``.
    - ``k - (n % k) = 2`` arrays of size ``n // k = 3``.

    >>> assigner = FoldAssigner(n_folds=3)
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

    def __init__(self, n_folds: int):
        self.n_folds = n_folds

    # --- Processing Methods -----------------------------------------------------------------------

    @overload
    def process(
        self,
        n_samples: int | None = None,
        mode: Literal["labels"] = "labels",
        seed: int = 0,
        **kwargs,
    ) -> FoldLabels: ...

    @overload
    def process(
        self,
        n_samples: int | None = None,
        mode: Literal["members"] = "members",
        seed: int = 0,
        **kwargs,
    ) -> FoldMembers: ...

    @set_random_state
    def process(
        self,
        n_samples: int | None = None,
        mode: Literal["labels", "members"] = "labels",
        seed: int = 0,
        **kwargs,
    ) -> Union[FoldLabels, FoldMembers]:
        """
        Implement the template method called in the base class `process` method.

        Arguments
        ---------
        n_samples : int
            Number of samples to assign to folds.
        mode : Literal["labels", "members"], default="labels"
            Return either the fold labels or the fold members.
        seed : int
            Seed for the random number generator.

        Returns
        -------
        fold_members: FoldMembers
            Indices of the samples contained in each fold. Number of sub-arrays: ``k``. Shapes:

            - ``n % k`` sub-arrays of size ``n // k + 1`` (to distribute the remainder's elements)
            - ``k - (n % k)`` sub-arrays of size ``n // k``.

        fold_labels : FoldLabels
            Fold labels assigned to each sample. Shape: ``(n_samples,)``.
        """
        assert n_samples is not None
        fold_members = self.assign(n_samples, self.n_folds)
        if mode == "members":
            return fold_members
        elif mode == "labels":
            return self.members_to_labels(fold_members)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @staticmethod
    @set_random_state
    def assign(n_samples: int, n_folds: int, seed: int = 0) -> FoldMembers:
        """
        Assign each sample to one fold.

        Arguments
        ---------
        n_samples : int
            See the argument `n_samples` in the `process` method.
        n_folds : int
            See the attribute `n_folds`.

        Returns
        -------
        fold_members: FoldMembers
            See the return value `fold_members` in the `process` method.

        Implementation
        --------------
        1. Extract the samples by strata.
        2. Within each strata, shuffle the samples to balance across folds the remaining variables
           have not been considered in stratification.
        3. Within the considered strata, distribute the shuffled samples into n_folds groups.
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
        fold_members = np.array_split(idx, n_folds)  # split samples into n_folds groups
        return fold_members

    # --- Conversion Methods -----------------------------------------------------------------------

    @staticmethod
    def labels_to_members(fold_labels: FoldLabels) -> FoldMembers:
        """
        Convert fold labels to fold members.

        Arguments
        ---------
        fold_labels : np.ndarray
            See the return value `fold_labels` in the `process` method.

        Returns
        -------
        fold_members : np.ndarray
            See the return value `fold_members` in the `process` method.
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
            See the return value `fold_members` in the `process` method.

        Returns
        -------
        fold_labels : np.ndarray
            See the return value `fold_labels` in the `process` method.
        """
        n_samples = np.max(fold_members) + 1
        fold_labels = np.full(n_samples, -1, dtype=np.int64)
        for i_fold, idx_samples in enumerate(fold_members):
            fold_labels[idx_samples] = i_fold
        return fold_labels

    @staticmethod
    def eval_min_count(n_samples: int, n_folds: int) -> int:
        """
        Evaluate the minimum count of samples assigned to each fold based on the total number of
        samples. Useful to determine the number of pseudo-trials fo form.

        Rule: `count_min = n_samples // n_folds` (see the `assign` method).

        Arguments
        ---------
        n_samples : int
            See the argument `n_samples` in the `process` method.
        n_folds : int
            See the attribute `n_folds`.

        Returns
        -------
        count : int
            Number of samples in each fold after assignment.
        """
        return n_samples // n_folds
