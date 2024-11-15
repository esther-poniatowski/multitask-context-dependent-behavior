#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.stratify` [module]

Classes
-------
`Stratifier`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# pylint: disable=useless-parent-delegation

from typing import overload, List, TypeAlias, Union, Any, Tuple, Dict, Literal

import numpy as np

from core.processors.base_processor import Processor


Strata: TypeAlias = np.ndarray[Tuple[Any], np.dtype[np.int64]]
"""Type alias for stratum labels."""

Features: TypeAlias = List[np.ndarray[Tuple[Any], np.dtype[Union[np.int64, np.float64, np.str_]]]]
"""Type alias for a list of feature arrays."""

FeatComb: TypeAlias = List[Tuple[Union[np.int64, np.float64, np.str_], ...]]
"""Type alias for the unique combinations of feature values which define each stratum."""


class Stratifier(Processor):
    """
    Divide a set of samples in strata (groups) based on combinations of features.

    Class Attributes
    ----------------
    UNASSIGNED_STRATUM : int
        Default value for unassigned stratum labels. See method `stratify`.

    Configuration Attributes
    ------------------------
    strata_def : FeatComb, optional
        Unique combinations of the feature values which define each single stratum.
        Length: ``n_strata``, number of strata labels.
        Shape of each element: ``(n_features,)``, number of features.
        The index of each tuple corresponds to the stratum label.
        If not provided, it is computed from the input features.

    Processing Arguments
    --------------------
    features : Features
        Features associated to the samples (e.g., task, attentional state, stimulus).
        Length: ``n_features``.
        Shape of each element (feature): ``(n_samples,)``.
        .. _features:
    return_comb : bool, optional
        If `True`, return the feature combinations used to define the strata.

    Returns
    -------
    strata : Strata
        Stratum labels of the samples. Shape: ``(n_samples,)``.
        .. _strata:
    strata_def : FeatComb, optional
        See the argument :ref:`strata_def`.

    Methods
    -------
    `stratify`
    `get_feature_types`
    `convert_strata_def`

    Examples
    --------
    Stratify three samples with integer, float, and string features:

    >>> features = [np.array([1, 1, 2], dtype=np.int64),
    ...             np.array([0.1, 0.1, 0.2], dtype=np.float64),
    ...             np.array(["A", "A", "B"], dtype=np.str_)]
    >>> stratifier = Stratifier()
    >>> strata, strata_def = stratifier.process(features=features)
    >>> print(strata)
    [0 0 1]
    >>> print(strata_def)

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    is_random: bool = False
    UNASSIGNED_STRATUM: int = -1

    def __init__(self, strata_def: FeatComb | None = None):
        super().__init__()
        if strata_def is not None:
            self._validate_configuration(strata_def)
        self.strata_def = strata_def

    # --- Validation Methods -----------------------------------------------------------------------

    def _validate_configuration(self, strata_def: FeatComb) -> None:
        """
        Validate the configuration parameters of the processor if provided.

        Raises
        ------
        ValueError
            If the feature combinations do not have the same length.
            If the feature combinations are not unique.
        """
        n_features = len(strata_def[0])
        if not all(len(comb) == n_features for comb in strata_def):
            raise ValueError("Unequal number of features in the feature combinations.")
        if len(strata_def) != len(set(strata_def)):
            raise ValueError("Non-unique feature combinations in `strata_def`.")

    def _pre_process(self, **input_data: Any) -> Dict[str, Any]:
        """
        Validate the features to consider for stratification.

        Raises
        ------
        ValueError
            If the number of samples is not equal across features.
            If the number of feature arrays does not match the number of features in combinations.
        """
        features = input_data.get("features", [])
        n_samples = [len(feat) for feat in features]
        if not all(n == n_samples[0] for n in n_samples):
            raise ValueError(f"Unequal number of samples across features: {n_samples}")
        if self.strata_def is not None:  # if provided
            n_features = len(self.strata_def[0])
            if len(features) != n_features:
                raise ValueError("Unequal number of features in `strata_def` and `features`.")
        return input_data

    # --- Processing Methods -----------------------------------------------------------------------

    @overload
    def _process(
        self, return_comb: Literal[True] = True, **input_data: Any
    ) -> Tuple[Strata, FeatComb]: ...

    @overload
    def _process(self, return_comb: Literal[False] = False, **input_data: Any) -> Strata: ...

    def _process(
        self, return_comb=False, **input_data: Any
    ) -> Union[Strata, Tuple[Strata, FeatComb]]:
        """Implement the template method called in the base class `process` method."""
        features = input_data["features"]
        if self.strata_def is None:
            strata_def = self.compute_combinations(features)
        else:
            strata_def = self.strata_def
        strata = self.stratify(features, strata_def)
        if return_comb:
            return strata, strata_def
        else:
            return strata

    def compute_combinations(self, features: Features) -> FeatComb:
        """
        Compute the unique combinations of features to define the strata.

        Output format: List of `n_strata` tuples, each containing the feature values which define a
        stratum. The index of each tuple corresponds to the stratum label.

        Arguments
        ---------
        features : Features
            See the argument :ref:`features`.
        feat_types : List[type]
            See the return value :ref:`feat_types`.

        Returns
        -------
        strata_def : FeatComb
            See the return value :ref:`strata_def`.

        Implementation
        --------------
        1. Identify the unique combinations of values among the features.
        2. Convert the unique combinations of features to their initial format.

        See Also
        --------
        :func:`numpy.column_stack`
            Stack 1-D arrays as columns into a 2-D array. If the feature arrays are of mixed types,
            the resulting stacked array will upcast data types to an array of `str`.
            Resulting shape (here): ``(n_samples, n_features)``.
        :func:`numpy.unique`
            Give the unique elements of an array, *sorted* in ascending or lexicographic order.
            Output: ``unique_feat``.
            Shape: ``(n_strata, n_features)``.
            Data type: ``np.str_`` or ``np.float64`` or ``np.int64``, depending on the types of the
            features. Prioritized order: ``np.str_`` -> ``np.float64`` -> ``np.int64``.
        """
        # Identify unit combinations of features
        feature_stack = np.column_stack(features)  # rows: samples, columns: features
        unique_feat = np.unique(feature_stack, axis=0)
        # Convert to appropriate types
        f_types = [feat.dtype.type for feat in features]
        strata_def = [tuple(tpe(val) for val, tpe in zip(comb, f_types)) for comb in unique_feat]
        return strata_def

    def stratify(self, features: Features, strata_def: FeatComb) -> Strata:
        """
        Compute stratum labels based on unique combinations of features.

        Arguments
        ---------
        features : Features
            See the argument :ref:`features`.
        strata_def : FeatComb
            See the attribute `strata_def`.

        Returns
        -------
        strata : Strata
            See the return value :ref:`strata`.

        Implementation
        --------------
        1. Stack the feature values to form a 2-D array of shape ``(n_samples, n_features)``.
        2. For each combination, assign a stratum label (index of the combination) to all the
           samples which match the combination.
        3. Check for invalid combinations in the input features, identified by a remaining value -1.
        """
        feature_stack = np.column_stack(features)
        strata = np.full(len(feature_stack), -1, dtype=np.int64)  # initialize with -1
        for i, comb in enumerate(strata_def):
            mask = np.all(feature_stack == comb, axis=1)
            strata[mask] = i
        if np.any(strata == self.UNASSIGNED_STRATUM):
            raise ValueError("Invalid combinations in `features`, not present in `strata_def`")
        return strata

    # --- Companion Methods ------------------------------------------------------------------------

    def get_stratum_indices(self, stratum_label: int, strata: Strata) -> np.ndarray:
        """
        Get the absolute indices of the samples in a specific stratum.

        Arguments
        ---------
        stratum_label : int
            Label of the stratum to extract.
        strata : Strata
            Stratum labels assigned to the samples. Shape: ``(n_samples,)``.

        Returns
        -------
        idx : np.ndarray
            Indices of the subset of samples from the stratum in the global dataset.

        Implementation
        --------------
        Find the indices of the trials in the stratum of interest:
        ``idx_in_stratum = np.where(strata == label)[0]``
        Extract the first (unique) element of the tuple since strata is one-dimensional.
        """
        return np.where(strata == stratum_label)[0]
