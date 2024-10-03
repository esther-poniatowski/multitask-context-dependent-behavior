#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.preprocess.stratify` [module]

Classes
-------
`Stratifier`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# pylint: disable=useless-parent-delegation

from typing import List, TypeAlias, Union, Any, Tuple, Optional, Dict

import numpy as np

from core.processors.base_processor import Processor


Strata: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for stratum labels."""

Features: TypeAlias = List[np.ndarray[Tuple[Any], np.dtype[Union[np.int64, np.float64, np.str_]]]]
"""Type alias for a list of feature arrays."""


class Stratifier(Processor):
    """
    Divide a set of samples in strata (groups) based on combinations of experimental features.

    Conventions for the documentation:

    - Attributes: Configuration parameters of the processor, passed to the *constructor*.
    - Arguments: Input data to process, passed to the `process` method (base class).
    - Returns: Output data after processing, returned by the `process` method (base class).

    Arguments
    ---------
    features : Features
        Features to consider to stratify the samples (e.g., task, context, stimulus).
        Length: ``n_features``.
        Shape of each element (feature): ``(n_samples,)``.
        .. _features:

    Returns
    -------
    strata : Strata
        Stratum labels of the samples. Shape: ``(n_samples,)``.
        .. _strata:
    feat_comb : Features, optional
        Unique combinations of the feature which define each single stratum.
        Length: ``n_features``.
        Shape of each element: ``(n_strata,)``, with ``n_strata`` the number of strata (labels).
        .. _feat_comb:

    Methods
    -------
    `stratify`
    `get_feature_types`
    `convert_feat_comb`

    Examples
    --------
    Stratify three samples with integer, float, and string features:

    >>> features = [np.array([1, 1, 2], dtype=np.int64),
    ...             np.array([0.1, 0.1, 0.2], dtype=np.float64),
    ...             np.array(["A", "A", "B"], dtype=np.str_)]
    >>> stratifier = Stratifier()
    >>> strata, feat_comb = stratifier.process(features=features)
    >>> print(strata)
    [0 0 1]
    >>> print(feat_comb)
    [array([1, 2]), array([0.1, 0.2]), array(['A', 'B'], dtype='<U1')]

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random: bool = False

    def __init__(self):
        super().__init__()  # no configuration parameters

    def _pre_process(
        self, features: Optional[Features] = None, **input_data: Any
    ) -> Dict[str, Any]:
        """
        Validate the features to be used for stratification.

        Raises
        ------
        ValueError
            If the number of samples is not equal across features.
        """
        assert features is not None
        n_samples = [len(feat) for feat in features]
        if not all(n == n_samples[0] for n in n_samples):
            raise ValueError(f"Unequal number of samples across features: {n_samples}")
        return input_data

    def _process(
        self, features: Optional[Features] = None, **input_data: Any
    ) -> Tuple[Strata, Features]:
        """Implement the template method called in the base class `process` method."""
        assert features is not None
        strata, feat_comb_raw = self.stratify(features)
        feat_types = self.get_feature_types(features)
        feat_comb = self.convert_feat_comb(feat_comb_raw, feat_types)
        return strata, feat_comb

    def stratify(self, features: Features) -> Tuple[Strata, np.ndarray]:
        """
        Compute stratum labels based on unique combinations of features.

        Arguments
        ---------
        features : Features
            See the argument :ref:`features`.

        Returns
        -------
        strata : Strata
            See the return value :ref:`strata`.
        feat_comb_raw : np.ndarray
            Unique combinations of the feature values, in a raw format.
            Shape: ``(n_strata, n_features)``.
            Data type: ``np.str_`` or ``np.float64`` or ``np.int64``, depending on the types of the
            features. Prioritized order: ``np.str_`` -> ``np.float64`` -> ``np.int64``.
            .. _feat_comb_raw:

        Implementation
        --------------
        1. Stack the feature values to identify unique combinations.
        2. Identify the unique combinations of the feature values.
        3. Assign a stratum label to each unique combination.
        4. Assign the stratum labels to the samples based on the values of their features.
        5. Convert the unique combinations of features to their initial format.

        See Also
        --------
        :func:`numpy.column_stack`
            Stack 1-D arrays as columns into a 2-D array. If the feature arrays are of mixed types,
            the resulting stacked array will upcast data types to an array of `str`.
            Resulting shape (here): ``(n_samples, n_features)``.
        :func:`numpy.unique`
            Find the unique elements of an array.
            Output: ``(feat_comb_raw, strata)``
            Parameter `return_inverse=True`: Return the *indices* of the unique combinations in the
            original array, which are used to assign the stratum labels to the samples.
        """
        feature_stack = np.column_stack(features)  # rows: samples, columns: features
        feat_comb_raw, strata = np.unique(feature_stack, axis=0, return_inverse=True)
        return strata, feat_comb_raw

    def get_feature_types(self, features: Features) -> List[type]:
        """
        Retrieve the types of the features.

        Arguments
        ---------
        features : Features
            See the argument :ref:`features`.

        Returns
        -------
        feat_types : List[type]
            Types of the features. Length: ``n_features``.
            .. _feat_types:
        """
        return [feat.dtype.type for feat in features]

    def convert_feat_comb(self, feat_comb_raw: np.ndarray, feat_types: List[type]) -> Features:
        """
        Convert the unique combinations of features to their initial format.

        Arguments
        ---------
        feat_comb_raw : np.ndarray
            See the return value :ref:`feat_comb_raw`.
        feat_types : List[type]
            See the return value :ref:`feat_types`.

        Returns
        -------
        feat_comb : Features
            See the return value :ref:`feat_comb`.
        """
        feat_comb: Features = [feat_comb_raw[:, i].astype(tpe) for i, tpe in enumerate(feat_types)]
        return feat_comb
