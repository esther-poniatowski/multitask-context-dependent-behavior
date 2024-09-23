#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.assign_folds` [module]

Classes
-------
:class:`FoldAssignerInputs`
:class:`FoldAssignerOutputs`
:class:`FoldAssigner`

Notes
-----
Folds correspond to subsets of samples (trials) used to train and test a model in cross-validation.

The class :class:`FoldAssigner` is only responsible of assigning each sample to one fold. Actual
splitting samples into training and testing sets is carried out in the :class:`CoordFold` class
itself based on these fold assignments (methods :meth:`get_train` and :meth:`get_test`). This allows
direct access to the samples in each set through the coordinate, without resorting to external
cross-validation tools.
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init
# pylint: disable=unused-argument

from dataclasses import dataclass
from typing import TypeAlias, Any, Tuple

import numpy as np

from core.processors.base_processor import Processor, ProcessorInput, ProcessorOutput
from core.processors.preprocess.stratify import Strata


Folds: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for fold assignments."""


@dataclass
class FoldAssignerInputs(ProcessorInput):
    """
    Dataclass for the inputs of the :class:`FoldAssigner` processor.

    Attributes
    ----------
    n_samples: int
        Number of samples to assign to folds.
    strata: Strata
        Labels of strata for stratified assignment. Shape: ``(n_samples,)``.
        If not provided or None, all samples are treated as belonging to a single stratum based on
        the number of samples `n_samples`.
    """

    n_samples: int
    strata: Strata

    def validate(self, k: int = 0, **config_params: Any) -> None:
        """
        Ensure consistency between both inputs:

        - Check that exactly one input is provided as argument (not both, nor none).
        - Set the default value for the missing attribute based on the other one.

        Arguments
        ---------
        k: int
            See :attr:`FoldAssigner.k`

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
        # Check that only one input is provided
        if self.n_samples is None and self.strata is None:
            raise ValueError("Missing arguments: provide either `n_samples` or `strata`.")
        if self.n_samples is not None and self.strata is not None:
            raise ValueError("Extra arguments: provide either `n_samples` or `strata`.")
        # Set the default value for the missing input
        if self.strata is None:
            self.strata = np.zeros(self.n_samples, dtype=np.int64)
        if self.n_samples is None:
            self.n_samples = self.strata.size
        # Check if the resulting number of samples is lower than the number of folds
        if self.n_samples < k:
            raise ValueError(f"n_samples: {self.n_samples} < k: {k}")


@dataclass
class FoldAssignerOutputs(ProcessorOutput):
    """
    Dataclass for the outputs of the :class:`FoldAssigner` processor.

    Attributes
    ----------
    folds: Folds
        Fold assignment for each sample. Shape: ``(n_samples,)``.
    """

    folds: Folds


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

    folds:
        Fold assignment for each sample. Shape: ``(n_samples,)``.

    Methods
    -------
    :meth:`assign`

    Warning
    -------
    Provide either `n_samples` or `strata` as input to the processor.

    Examples
    --------
    Assign 10 samples to 3 folds:

    >>> assigner = FoldAssigner(k=3)
    >>> folds = assigner.process(n_samples=10)
    >>> print(folds)
    [0 0 1 1 2 2 0 1 2 0]

    Use stratified fold assignment:

    >>> strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    >>> folds = assigner.process(strata=strata)
    >>> print(folds)
    [0 0 1 1 2 2]

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors. See definition of class-level attributes and template
        methods.
    """

    config_params = ("k",)
    input_dataclass = FoldAssignerInputs
    output_dataclass = FoldAssignerOutputs
    is_random = True

    def __init__(self, k: int):
        super().__init__(k=k)

    def _process(self, strata: Strata = FoldAssignerInputs.strata, **input_data: Any) -> Folds:
        """Implement the template method called in the base class :meth:`process` method."""
        folds = self.assign(strata)
        return folds

    def assign(self, strata: Strata) -> Folds:
        """
        Assign folds to each sample based on strata labels.

        Arguments
        ---------
        strata: Strata
            See :attr:`FoldAssignerInputs.strata`.

        Returns
        -------
        Folds
            See :attr:`FoldAssignerOutputs.folds`.

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
        folds = np.zeros(strata.size, dtype=np.int64)  # initialize folds
        for stratum in np.unique(strata):
            idx_stratum = np.where(strata == stratum)[0]  # indices of samples in the stratum
            np.random.shuffle(idx_stratum)  # shuffle samples before splitting
            split_indices = np.array_split(idx_stratum, self.k)  # split samples into k groups
            for i_fold, idx_samples in enumerate(split_indices):
                folds[idx_samples] = i_fold
        return folds
