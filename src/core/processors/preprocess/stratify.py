#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.stratify` [module]

Classes
-------
:class:`Stratifier`
"""
from types import MappingProxyType
from typing import List, TypeAlias, Union, Dict, Any

import numpy as np
import numpy.typing as npt

from core.processors.base import Processor

Strata: TypeAlias = npt.NDArray[np.int64]
"""Type alias for stratum labels."""
Features = List[npt.NDArray[Union[np.int64, np.float64, np.str_]]]
"""Type alias for a list of feature arrays."""


class Stratifier(Processor):
    """
    Divide a set of samples in strata (groups) based on combinations of experimental features.

    Class Attributes
    ----------------
    See :class:`Processor` for inherited class attributes.
    valid_types : tuple
        Valid NumPy data types (dtype) for the feature arrays (int64, float64, str_).

    Attributes
    ----------
    features : list of npt.NDArray
        Features to consider to stratify the samples (e.g., task, context, stimulus).
        Length: ``n_features``.
        Shape of each element (feature): ``(n_samples,)``.
    strata : npt.NDArray[np.int64]
        Stratum labels of the samples. Shape: ``(n_samples,)``.

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
    >>> stratifier = Stratifier()
    >>> stratifier.process(features=features)
    >>> strata = stratifier.strata  # access the cached strata via the property
    >>> print(strata)
    [0 0 1]

    Update the features and recompute strata:

    >>> new_features = [np.array([1, 2, 2], dtype=np.int64),
    ...                 np.array([0.2, 0.2, 0.2], dtype=np.float64),
    ...                 np.array(["X", "Y", "Y"], dtype=np.str_)]
    >>> stratifier.process(features=new_features) # automatically resets strata cache
    >>> print(stratifier.strata)
    [0 1 1]

    Implementation
    --------------
    Private attributes are used to enforce control and validation: `_features`, `_strata`.
    Those private attributes are accessed via the corresponding public properties (with no
    underscore) which are automatically created by the metaclass `ProcessorMeta`.
    """

    config_attrs = ()
    input_attrs = ("features",)
    output_attrs = ("strata",)
    proc_data_empty = MappingProxyType({"features": [], "strata": np.array([], dtype=np.int64)})
    valid_types = (np.int64, np.float64, np.str_)

    def __init__(self):
        super().__init__()  # call the parent class constructor (no config attributes)

    def _validate_inputs(self, **input_data) -> None:
        """Implement the template method called in the base class :meth:`process` method."""
        self._validate_features(input_data["features"])

    def _validate_features(self, features: Features) -> None:
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
            raise ValueError(f"Feature(s) not numpy arrays: {features}")
        types = [feat.dtype.type for feat in features]
        if not all(tpe in self.valid_types for tpe in types):
            raise ValueError(f"Invalid feature types: {types}")
        dims = [feat.ndim for feat in features]
        if not all(dim == 1 for dim in dims):
            raise ValueError(f"Invalid feature dimensions: {dims}")
        n_samples = [len(feat) for feat in features]
        if not all(n == n_samples[0] for n in n_samples):
            raise ValueError(f"Unequal number of samples across features: {n_samples}")

    def _process(self) -> Dict[str, Any]:
        """Implement the template method called in the base class :meth:`process` method."""
        strata = self.stratify()
        return {"strata": strata}

    def stratify(self) -> Strata:
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
