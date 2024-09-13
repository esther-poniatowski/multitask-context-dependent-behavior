#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.assign_folds` [module]

Classes
-------
:class:`FoldsAssigner`

Notes
-----
Folds correspond to subsets of samples (trials) used to train and test a model in cross-validation.

The class :class:`FoldsAssigner` is only responsible of assigning each sample to one fold. Actual
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
from typing import Optional, TypeAlias

import numpy as np
import numpy.typing as npt


Strata: TypeAlias = npt.NDArray[np.int64]
"""Type alias for stratum labels."""
Folds: TypeAlias = npt.NDArray[np.int64]
"""Type alias for fold assignments."""


class FoldsAssigner:
    """
    Assign samples (trials) to folds for cross-validation.

    Attributes
    ----------
    k : int
        Number of folds in which the samples will be divided. Read-only.
    n_samples : int
        Number of samples to assign to folds.
    strata : npt.NDArray[np.int64]
        Labels of strata for stratified assignment. Shape: ``(n_samples,)``.
        If None, all samples are treated as belonging to a single stratum.
    folds : npt.NDArray[np.int64]
        Fold assignment for each sample. Shape: ``(n_samples,)``.
    seed : int, default=0
        Random state for reproducibility in shuffling samples before fold assignment.

    Methods
    -------
    :meth:`_validate_n_samples`
    :meth:`_validate_strata`
    :meth:`_create_strata`
    :meth:`set_seed`
    :meth:`assign`

    Examples
    --------
    Assign 10 samples to 3 folds:

    >>> assigner = FoldsAssigner(k=3, n_samples=10)
    >>> print(assigner.folds)
    [0 0 1 1 2 2 0 1 2 0]

    Update to 6 samples and reassign the folds:

    >>> assigner.n_samples = 6
    >>> print(assigner.folds)
    [0 0 1 1 2 2]

    Use stratified fold assignment:

    >>> strata = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    >>> assigner.strata = strata
    >>> print(assigner.folds)
    [0 0 1 1 2 2]

    Implementation
    --------------
    Private attributes are used to enforce control and validation:

    - `_k`: Accessed via the property `k`, set during instantiation. It is read-only to ensure that
      the same parameter will be used consistently for various sets of samples.
    - `_n_samples`: Accessed via the property `n_samples`, set by the property setter. It validates
      the input number of samples and resets the cache `_folds` if the number of samples is updated.
    - `_strata`: Accessed via the property `strata` and set by the property setter. It validates the
      input strata and resets the cache `_folds` if strata are updated.
    - `_folds`: Accessed and set via the property `folds`. It computes this attribute lazily on
      first access and caches it for subsequent accesses.
    """

    def __init__(
        self,
        k: int,
        n_samples: Optional[int] = None,
        strata: Optional[Strata] = None,
        seed: int = 0,
    ):
        # Set simple attributes
        self._k = k  # read-only
        self.seed = seed
        # Initialize the cache
        self._folds: Optional[Folds] = None
        # Declare types for private attributes set by property setters
        self._strata: Strata
        self._n_samples: int
        # Set attributes based on input
        if strata is not None:
            if n_samples is not None and n_samples != len(strata):
                raise ValueError(f"Mismatch: len(strata): {len(strata)} != n_samples: {n_samples}")
            self.strata = strata  # set n_samples automatically
        elif n_samples is not None:
            self.n_samples = n_samples  # set strata automatically
        else:
            raise ValueError("Missing argument: neither n_samples nor strata.")

    @property
    def k(self) -> int:
        """Read-only property for `k` to ensure it cannot be modified after instantiation."""
        return self._k

    @property
    def n_samples(self) -> int:
        """Access to the private attribute :attr:`_n_samples`."""
        return self._n_samples

    @n_samples.setter
    def n_samples(self, n: int) -> None:
        """Validate and set `_n_samples`, create new `_strata`, reset the cache `_folds`.

        Create default strata labels, treating all samples as belonging to a single stratum by
        setting `_strata` to an array of zeros with the same length as the number of samples.
        """
        self._validate_n_samples(n)
        self._n_samples = n
        self._strata = np.zeros(n, dtype=np.int64)
        self._folds = None

    @property
    def strata(self) -> Strata:
        """Access to the private attribute `_strata`."""
        return self._strata

    @strata.setter
    def strata(self, new_strata: Strata) -> None:
        """Validate and set `_strata`, set `_n_samples` accordingly, reset the cache `_folds`.

        Set the number of samples to the length of the strata array..
        """
        self._validate_strata(new_strata)
        self._strata = new_strata
        self._n_samples = len(new_strata)
        self._folds = None

    @property
    def folds(self) -> Folds:
        """Access the cache `_folds`, compute it if empty."""
        if self._folds is None:
            self._folds = self.assign()
        return self._folds

    def _validate_n_samples(self, n: int) -> None:
        """
        Validate the number of samples compared to the number of folds.

        Raises
        ------
        ValueError
            If the number of samples is lower than the number of folds.
        """
        if n < self._k:
            raise ValueError(f"n_samples: {n} < k: {self._k}")

    def _validate_strata(self, strata: Strata) -> None:
        """
        Validate the strata used for stratification if provided.

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
        if len(strata) < self._k:
            raise ValueError(f"len(strata): {len(strata)} < k: {self._k}")

    def set_seed(self) -> None:
        """
        Set the random seed for reproducibility.

        See Also
        --------
        :func:`np.random.seed`: Set the random seed for reproducibility.
        """
        np.random.seed(self.seed)

    def assign(self) -> Folds:
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
