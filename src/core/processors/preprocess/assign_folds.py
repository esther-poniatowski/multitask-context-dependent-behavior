#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.assign_folds` [module]

Classes
-------
:class:`FoldAssigner`

Notes
-----
Folds correspond to subsets of samples (trials) used to train and test a model in cross-validation.

The class :class:`FoldAssigner` is only responsible of assigning each sample to one fold. Actual
splitting samples into training and testing sets is carried out in the :class:`CoordFold` class
itself based on these fold assignments (methods :meth:`get_train` and :meth:`get_test`). This allows
direct access to the samples in each set through the coordinate, without resorting to external
cross-validation tools.

Warning
-------
For data analysis:

- Assign trials to folds *by unit*, before constructing pseudo-trials via hierarchical bootstrap.
  This ensures that trials are combined within each fold and prevents data leakage across folds.
- Use *stratified* assignment by condition (task, context, stimulus, error) to balance trial types
  across folds.

Implementation
--------------
Procedure for fold assignment:

1. Keep valid trial (if necessary).
2. Split samples by condition (task, context, stimulus, error).
2. For each condition, shuffle the trials and distribute them in k groups.
3. Assign each trial to a fold by selecting the corresponding group.

Shuffling before splitting aims to balance across folds the task variables which have not been
considered in stratification (i.e. positional information: recording number, block number, slot
number). This prevents models to capture misleading temporal drift in neuronal activity.
"""
# Disable error codes for attributes which are not detected by the type checker:
# - Configuration attributes are defined by the base class constructor.
# - Public properties for internal attributes are defined in the metaclass.
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member

from types import MappingProxyType
from typing import Dict, TypeAlias, Any, Tuple

import numpy as np

from core.processors.base import Processor


Strata: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for stratum labels."""

Folds: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for fold assignments."""


class FoldAssigner(Processor):
    """
    Assign samples (trials) to folds for cross-validation.

    Attributes
    ----------
    k: int
        Number of folds in which the samples will be divided. Read-only.
    n_samples: int
        Number of samples to assign to folds.
        If not provided or None, the number of samples is inferred from the length of `strata`.
    strata: np.ndarray[Tuple[Any], np.dtype[np.int64]]
        Labels of strata for stratified assignment. Shape: ``(n_samples,)``.
        If not provided or None, all samples are treated as belonging to a single stratum based on
        the number of samples `n_samples`.
    folds: np.ndarray[Tuple[Any], np.dtype[np.int64]]
        Fold assignment for each sample. Shape: ``(n_samples,)``.

    Methods
    -------
    :meth:`_validate_n_samples`
    :meth:`_validate_strata`
    :meth:`assign`

    Warning
    -------
    Provide either `n_samples` or `strata` as input to the processor.

    Examples
    --------
    Assign 10 samples to 3 folds:

    >>> assigner = FoldAssigner(k=3)
    >>> assigner.process(n_samples=10)
    >>> print(assigner.folds)
    [0 0 1 1 2 2 0 1 2 0]

    Update to 6 samples and reassign the folds:

    >>> assigner.process(n_samples=6)
    >>> print(assigner.folds)
    [0 0 1 1 2 2]

    Use stratified fold assignment:

    >>> strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    >>> assigner.process(strata=strata)
    >>> print(assigner.folds)
    [0 0 1 1 2 2]

    Implementation
    --------------
    Private attributes used to enforce control and validation: `_n_samples`, `_folds`, `_strata`.
    """

    config_attrs = ("k",)
    input_attrs = ("n_samples", "strata")
    output_attrs = ("folds",)
    proc_data_empty = MappingProxyType(
        {
            "n_samples": 0,
            "strata": np.array([], dtype=np.int64),
            "folds": np.array([], dtype=np.int64),
        }
    )

    def __init__(self, k: int):
        super().__init__(k=k)

    def _validate(self, **input_data: Any) -> None:
        """
        Implement the template method called in the base class :meth:`process` method.

        Raises
        ------
        ValueError
            If both `n_samples` and `strata` are missing.
            If both `n_samples` and `strata` are provided.
            If any of `n_samples` or `strata` is provided and invalid.
        """
        # Get input data (None if not provided)
        n_samples = input_data.get("n_samples")
        strata = input_data.get("strata")
        # If both missing or both provided, raise error
        if n_samples is None and strata is None:
            raise ValueError("Missing arguments: provide either `n_samples` or `strata`.")
        elif n_samples is not None and strata is not None:
            raise ValueError("Extra arguments: provide either `n_samples` and `strata`.")
        # Specific validation if provided
        elif strata is not None:
            self._validate_strata(strata)
        elif n_samples is not None:
            self._validate_n_samples(n_samples)

    def _validate_n_samples(self, n: int) -> None:
        """
        Validate the argument `n_samples` (number of samples) compared to the number of folds.

        Raises
        ------
        ValueError
            If the number of samples is lower than the number of folds.
        """
        if n < self.k:
            raise ValueError(f"n_samples: {n} < k: {self.k}")

    def _validate_strata(self, strata: Strata) -> None:
        """
        Validate the argument `strata` (stratum labels) based on its structure and content.

        Raises
        ------
        ValueError
            If the strata is not 1D.
            If the strata dtype is not int64.
            If the number of samples in the strata is lower than the number of folds.
        """
        if strata.ndim != 1:
            raise ValueError(f"Invalid strata dimension: {strata.ndim}")
        if not np.issubdtype(strata.dtype, np.integer):
            raise ValueError(f"Invalid strata dtype: {strata.dtype}")
        if len(strata) < self.k:
            raise ValueError(f"len(strata): {len(strata)} < k: {self.k}")

    def _default(self, **input_data: Any) -> Dict[str, Any]:
        """
        Implement the template method called in the base class :meth:`process` method.

        Rules to ensure consistency between `n_samples` and `strata` if one of them is missing:

        - `strata`: All samples are treated as belonging to a single stratum (single label 0).
        - `n_samples`: Equal to the length of `strata`.

        Returns
        -------
        input_data: Dict[str, Any]
            Input data with default values set for missing arguments.

        Notes
        -----
        Since this method is called after validation, the input data is guaranteed to contain
        exactly one of `n_samples` or `strata` (not both, nor none).
        """
        n_samples = input_data.get("n_samples")
        strata = input_data.get("strata")
        # If only strata provided, set n_samples
        if strata is not None:  # n_samples is None
            input_data["n_samples"] = len(strata)
        # If only n_samples provided, set default strata
        elif n_samples is not None:  # strata is None
            input_data["strata"] = np.zeros(n_samples, dtype=np.int64)
        return input_data

    def _process(self) -> Dict[str, Any]:
        """
        Implement the template method called in the base class :meth:`process` method.

        Returns
        -------
        folds: np.ndarray[Tuple[Any], np.dtype[np.int64]]
            See :attr:`folds`.
        """
        folds = self.assign()
        return {"folds": folds}

    def assign(self) -> Folds:
        """
        Assign folds to trials, stratified by condition.

        Returns
        -------
        folds: np.ndarray[Tuple[Any], np.dtype[np.int64]]
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
        self.set_random_state()
        folds = np.zeros(self.n_samples, dtype=np.int64)  # initialize folds
        for stratum in np.unique(self.strata):
            idx_stratum = np.where(self.strata == stratum)[0]  # indices of samples in the stratum
            np.random.shuffle(idx_stratum)  # shuffle samples before splitting
            split_indices = np.array_split(idx_stratum, self.k)  # split samples into k groups
            for i_fold, idx_samples in enumerate(split_indices):
                folds[idx_samples] = i_fold
        return folds
