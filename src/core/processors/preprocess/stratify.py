#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.stratify` [module]

Classes
-------
:class:`Stratifier`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# mypy: disable-error-code="attr-defined"
# pylint: disable=no-member
# pylint: disable=attribute-defined-outside-init

from types import MappingProxyType
from typing import List, TypeAlias, Union, Any, Tuple

import numpy as np

from core.processors.base import Processor
from utils.misc.arrays import create_empty_array


Strata: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for stratum labels."""

Features = List[np.ndarray[Tuple[Any], np.dtype[Union[np.int64, np.float64, np.str_]]]]
"""Type alias for a list of feature arrays."""


class Stratifier(Processor):
    """
    Divide a set of samples in strata (groups) based on combinations of experimental features.

    Class Attributes
    ----------------
    valid_types: tuple
        Valid NumPy data types (dtype) for the feature arrays (int64, float64, str_).

    Attributes
    ----------
    features: List[np.ndarray[Tuple[Any], np.dtype[Union[np.int64, np.float64, np.str_]]]]
        Features to consider to stratify the samples (e.g., task, context, stimulus).
        Length: ``n_features``.
        Shape of each element (feature): ``(n_samples,)``.
    strata: np.ndarray[Tuple[Any], np.dtype[np.int64]]
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

    See Also
    --------
    :class:`core.processors.preprocess.base.Processor`
        Base class for all processors. See definition of class-level attributes and template
        methods.
    """

    config_attrs = ()
    input_attrs = ("features",)
    output_attrs = ("strata",)
    empty_data = MappingProxyType({"features": [], "strata": create_empty_array(1, np.int64)})
    valid_types = (np.int64, np.float64, np.str_)

    def __init__(self):
        super().__init__()  # call the parent class constructor (no config attributes)

    def _validate(self, **input_data) -> None:
        """Implement the template method called in the base class :meth:`process` method."""
        self._validate_features(input_data["features"])

    def _process(self) -> None:
        """Implement the template method called in the base class :meth:`process` method."""
        self.stratify()

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

    def stratify(self) -> None:
        """
        Compute stratum labels based on unique combinations of features.

        Important
        ---------
        Update the attribute `strata` with the computed stratum labels.

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
            Resulting shape (here): ``(n_samples, n_features)``.
        :func:`numpy.unique`
            Find the unique elements of an array.
            Output: ``(unique_combinations, strata)``
            Parameter `return_inverse=True`: Return the indices of the unique combinations in the
            original array, which are used to assign the stratum labels to the samples.
        """
        feature_stack = np.column_stack(self.features)  # rows: samples, columns: features
        _, strata = np.unique(feature_stack, axis=0, return_inverse=True)  # shape: (n_samples,)
        self.strata = strata  # cache strata labels
