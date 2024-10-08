#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.map_indices` [module]

Classes
-------
`IndexMapper`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# pylint: disable=useless-parent-delegation
# pylint: disable=no-member
# mypy: disable-error-code="attr-defined"

from typing import TypeAlias, Any, Tuple, List

import numpy as np

from core.processors.base_processor import Processor


Indices: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for indices."""


class IndexMapper(Processor):
    """
    Map relative indices to absolute indices in a global data set.

    Conventions for the documentation:

    - Attributes: Configuration parameters of the processor, passed to the *constructor*.
    - Arguments: Input data to process, passed to the `process` method (base class).
    - Returns: Output data after processing, returned by the `process` method (base class).

    Arguments
    ---------
    idx_absolute : Indices
        Indices of a subset of trials within a global data set ("absolute" indices).
        Shape: ``(n_samples,)``, with ``n_samples`` the number of samples in the *subset*.
        Values: Comprised between ``0`` and ``n_tot - 1``, with ``n_tot`` the total number of
        samples in the global data set.
        .. _idx_absolute:
    idx_relative : Indices
        Indices of samples within the subset ("relative" indices).
        Shape: ``(n,)``, with ``n`` an arbitrary number of selected samples.
        Values: Comprised between ``0`` and ``n_samples - 1``.
        .. _idx_relative:

    Returns
    -------
    idx_mapped : Indices
        Indices of the samples indexed in `idx_relative` within the global data set. Each index in
        `idx_relative` has been replaced by the corresponding index in `idx_absolute`.
        Shape: ``(n,)``, matching the shape of `idx_relative`.
        Values: Indices from `idx_absolute`.
        .. _idx_mapped:

    Methods
    -------
    `relative_to_absolute`

    Examples
    --------
    Transpose relative indices to absolute indices in the global dataset:

    >>> index_mapper = IndexMapper(strata)
    >>> data = np.arange(10) # 10 samples in total
    >>> idx_absolute = np.where(data % 2 == 0)[0] # 5 selected samples
    >>> print(idx_absolute)
    [0 2 4 6 8]
    >>> idx_relative = np.repeat(np.arange(3), 2) # 3 samples selected twice
    >>> print(idx_relative)
    [0 0 1 1 2 2]
    >>> idx_mapped = index_mapper.process(idx_absolute=idx_absolute, idx_relative=idx_relative)
    >>> print(idx_mapped)
    [0 0 2 2 4 4]

    Explanation:

    - idx_relative [0, 0] maps to idx_absolute[0] = 0
    - idx_relative [1, 1] maps to idx_absolute[1] = 2
    - idx_relative [2, 2] maps to idx_absolute[2] = 4

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random: bool = False

    def __init__(self):
        super().__init__()

    def _process(self, **input_data: Any) -> Indices:
        """Implement the template method called in the base class `process` method."""
        idx_absolute = input_data["idx_absolute"]
        idx_relative = input_data["idx_relative"]
        return self.relative_to_absolute(idx_absolute, idx_relative)

    def relative_to_absolute(self, idx_absolute: Indices, idx_relative: Indices) -> Indices:
        """
        Transpose relative indices to absolute indices in the global dataset.

        Arguments
        ---------
        idx_absolute : Indices
            See the attribute `idx_absolute`.
        idx_relative : Indices
            See the attribute `idx_relative`.

        Returns
        -------
        idx_mapped : np.ndarray
            See the return value `idx_mapped`.

        Implementation
        --------------
        Replace each relative index by the corresponding absolute index:

        ``idx_mapped = idx_absolute[idx_relative]``

        This generates an array with the same shape as `idx_relative`. Each value is picked from the
        array `idx_absolute` (indices in the global data set) at the index specified in
        `idx_relative` (which indicate their positions among the subset).
        """
        return idx_absolute[idx_relative]

    # --- Companion Methods ------------------------------------------------------------------------

    def get_stratum_indices(self, stratum_label: int) -> Indices:
        """
        Get the absolute indices of the samples in a specific stratum.

        Arguments
        ---------
        stratum_label : int
            See the argument :ref:`stratum_label`.

        Returns
        -------
        idx_absolute : np.ndarray
            Absolute indices of the samples in the global dataset.

        Implementation
        --------------
        Find the indices of the trials in the stratum of interest:
        ``idx_in_stratum = np.where(strata == label)[0]``
        Extract the first (unique) element of the tuple since strata is one-dimensional.
        """
        return np.where(self.strata == stratum_label)[0]

    def gather_across_strata(self, data_relative: List[np.ndarray]) -> np.ndarray:
        """
        Gather data obtained for distinct subset of samples (strata) in a global structure
        corresponding to the entire data set.

        Arguments
        ---------
        data_relative : List[np.ndarray]
            Data obtained for each stratum. Length: `n_strata`.
            Each element contains data for the samples in the corresponding stratum, ordered
            by the relative index of the samples within the stratum.
            All the data arrays must have the same number of dimensions and the same shape, except
            for the first dimension which is the number of samples in the stratum.

        Returns
        -------
        data_absolute : np.ndarray
            Data gathered for all the samples across distinct strata.
            Shape: ``(n_samples, ...)``. The other dimensions depend on the nature of the results.
        """
        n_samples = self.strata.size
        other_dims = data_relative[0].shape[1:]
        shape = (n_samples,) + other_dims
        dtype = data_relative[0].dtype
        data_absolute = np.empty(shape, dtype=dtype)
        for stratum_label, results_stratum in enumerate(data_relative):
            idx_absolute = self.get_stratum_indices(stratum_label)
            data_absolute[idx_absolute] = results_stratum
        return data_absolute
