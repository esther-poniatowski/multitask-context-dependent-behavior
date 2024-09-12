#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.pipelines.preprocess.stratify` [module]

Classes
-------
:class:`Stratifier`
"""
from typing import List, Optional

import numpy as np
import numpy.typing as npt


class Stratifier:
    """
    Divide a set of samples in strata (groups) based on combinations of experimental features.

    Class Attributes
    ----------------
    valid_types : tuple
        Valid types for the feature values.

    Attributes
    ----------
    features : list of npt.NDArray
        Features to consider to stratify the samples (e.g., task, context, stimulus).
        Length: ``n_features``.
        Shape of each element (feature): ``(n_samples,)``.
    _strata : npt.NDArray[np.int64]
        (Internal attribute) Stratum labels for each sample. Shape: ``(n_samples,)``.
        Computed lazily on first access and cached for subsequent accesses.
    strata : npt.NDArray[np.int64]
        (Property) Access to the stratum labels of the samples.

    Methods
    -------
    :meth:`_validate_features`

    :meth:`stratify`

    Examples
    --------
    Stratify three samples with integer, float, and string features:

    >>> features = [np.array([1, 1, 2], dtype=np.int64),
    ...             np.array([0.1, 0.1, 0.2], dtype=np.float64),
    ...             np.array(["A", "A", "B"], dtype=np.str_)]
    >>> stratifier = Stratifier(features)
    >>> strata = stratifier.strata  # access the cached strata via the property
    >>> print(strata)
    [0 0 1]

    """

    valid_types = (np.int64, np.float64, np.str_)

    def __init__(self, features: List[npt.NDArray]):
        self._validate_features(features)
        self.features = features
        self._strata: Optional[npt.NDArray[np.int64]] = None

    @property
    def strata(self) -> npt.NDArray[np.int64]:
        """Stratum labels for each sample."""
        if self._strata is None:
            self._strata = self.stratify()
        return self._strata

    def _validate_features(self, features: List[npt.NDArray]) -> None:
        """
        Validate the features to be used for stratification.

        Raises
        ------
        ValueError
            If the features are not NumPy arrays.
            If the feature types are not valid.
            If the feature dimensions are not 1.
            If the number of samples in each feature array is not equal.
        """
        if not all(isinstance(feat, np.ndarray) for feat in features):
            raise ValueError("[ERROR] All features must be NumPy arrays.")
        types = [feat.dtype.type for feat in features]
        if not all(tpe in self.valid_types for tpe in types):
            raise ValueError(f"[ERROR] Invalid feature types: {types}")
        dims = [feat.ndim for feat in features]
        if not all(dim == 1 for dim in dims):
            raise ValueError(f"[ERROR] Invalid feature dimensions: {dims}")
        n_samples = [len(feat) for feat in features]
        if not all(n == n_samples[0] for n in n_samples):
            raise ValueError(f"[ERROR] Unequal number of samples across features: {n_samples}")

    def stratify(self) -> npt.NDArray[np.int64]:
        """
        Compute stratum labels based on unique combinations of features.

        Returns
        -------
        strata: npt.NDArray[np.int64]
            See :attr:`strata`.

        Implementation
        --------------

        1. Stack the feature values to identify unique combinations.
        2. Identify the unique combinations of the feature values.
        3. Assign a stratum label to each unique combination.
        4. Assign the stratum labels to the samples based on the values of their features.

        See Also
        --------
        :func:`numpy.column_stack`
            Stack 1-D arrays as columns into a 2-D array. If the feature arrays are of mixed types,
            the resulting stacked array will upcast data types to an array of `str`.
        :func:`numpy.unique`
            Find the unique elements of an array.
            Output: ``(unique_combinations, strata)``
            Parameter `return_inverse=True`: Return the indices of the unique combinations in the
            original array, which are used to assign the stratum labels to the samples.
        """
        feature_stack = np.column_stack(self.features)  # shape: (n_samples, n_features)
        _, strata = np.unique(feature_stack, axis=0, return_inverse=True)  # shape: (n_samples,)
        return strata
