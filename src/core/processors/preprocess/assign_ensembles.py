#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.assign_ensembles` [module]

Classes
-------
EnsembleAssigner

Notes
-----
Ensembles correspond to subsets of units (neurons) in each brain area.

For comparative analysis between brain areas with varying neuron populations, it is necessary to
homogenize the number of neurons across different models. The common number of neurons retained for
all models is imposed by the minimal number of neurons across all the brain areas. In order to
leverage the full dat set for areas (i.e. to encompass all the neurons in each area), multiple
models have to be fitted for ensembles in areas with larger neuron populations.
"""

from typing import TypeAlias, Any, Tuple, Optional

import numpy as np

from core.processors.base_processor import Processor, set_random_state


Ensembles: TypeAlias = np.ndarray[Tuple[Any, Any], np.dtype[np.int64]]
"""Type alias for ensemble assignments."""


class EnsembleAssigner(Processor):
    """
    Assign units (neurons) to ensembles ('batches' for data analysis).

    Attributes
    ----------
    ensemble_size : int
        Number of units included in each ensemble.
    n_ensembles_max : int, optional
        Maximum number of ensembles to generate.

    Methods
    -------
    `process` (required)
    `assign` (static)
    `eval_n_ensembles` (static)
    `limit_ensembles` (static)

    Examples
    --------
    Assign 10 units to ensembles of size of 4:

    >>> assigner = EnsembleAssigner(ensemble_size=4)
    >>> ensembles = assigner.process(n_units=10)
    >>> print(ensembles)
    [[0 7 4 3]
     [2 8 6 1]
     [5 9 0 6]]

    Explanation:

    - 3 ensembles are generated so that each unit is included in at least one ensemble.
    - The first two ensembles are mutually exclusive, while the last ensemble includes the leftover
      units and picks remaining units from the previous ensembles.

    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
    `core.processors.base_processor.set_random_state`
    """

    def __init__(self, ensemble_size: int, n_ensembles_max: Optional[int] = None):
        self.ensemble_size = ensemble_size
        self.n_ensembles_max = n_ensembles_max

    @set_random_state
    def process(self, n_units: int | None = None, **kwargs) -> Ensembles:
        """
        Implement the abstract method of the base class `Processor`.

        Arguments
        ---------
        n_units : int
            Number of units to assign to ensembles.

        Returns
        -------
        ensembles : Ensembles
            Ensemble assignments, containing the indices of the units forming each ensemble.
            Shape: ``(n_ensembles, ensemble_size)``.
        """
        assert n_units is not None
        self.validate(n_units)
        ensembles = self.assign(n_units, self.ensemble_size)
        if self.n_ensembles_max is not None:
            ensembles = self.limit_ensembles(ensembles, self.n_ensembles_max)
        return ensembles

    # --- Pre-processing methods -------------------------------------------------------------------

    def validate(self, n_units: int) -> None:
        """
        Validate the argument `n_units` (number of units) compared to the ensemble size.

        Raises
        ------
        ValueError
            If the number of units is lower than the ensemble size.
        """
        if n_units < self.ensemble_size:
            raise ValueError(f"n_units: {n_units} < ensemble_size: {self.ensemble_size}")

    # --- Processing methods -----------------------------------------------------------------------

    @staticmethod
    def eval_n_ensembles(n_units: int, ensemble_size: int, n_ensembles_max: int) -> int:
        """
        Determine the number of ensembles to generate.

        Arguments
        ---------
        n_units : int
            See the argument :ref:`n_units` in the `process` method.
        ensemble_size : int
            See the configuration attribute `ensemble_size`.
        n_ensembles_max : int
            See the configuration attribute `n_ensembles_max`.

        Returns
        -------
        n_ensembles : int
            Number of ensembles to generate based on the number of units and the ensemble size.

        Notes
        -----
        Rules to determine the number of ensembles from the number of units and the target ensemble
        size:

        - If the number of units is equal to the ensemble size, then all units form a single
          ensemble. Thus, ``n_ensembles = 1``.
        - If the number of units is greater than the ensemble size, then units are gathered in
          groups of the target ensemble size as long as possible, and a last ensemble is created by
          gathering the remaining units and picking among the units which have already been included
          in another ensemble. If the number of units is a perfect multiple of the size of an
          ensemble, the number of ensembles is ``n_ensembles = n_units // ensemble_size``. If
          the number of units is not a perfect multiple, the number of ensembles is computed as the
          ceil of the division of the number of units by the ensemble size, i.e. the number is equal
          to ``n_ensembles = n_units // ensemble_size + 1``.
        - If the maximum number of ensembles is imposed as a configuration parameter and the
          pre-computed number of ensembles exceeds this maximum, then the number of ensembles is
          reset to the maximum.

        Warning
        -------
        This method is not used for the computation of the ensembles. It can be used to assess the
        resulting number of ensembles that will be generated by the processor, for instance to
        initialize data structures with appropriate dimensions.

        See Also
        --------
        :func:`numpy.ceil`: Round up to the nearest integer (floating point output).
        """
        n_ensembles = np.ceil(n_units / ensemble_size).astype(int)
        if n_ensembles_max is not None and n_ensembles > n_ensembles_max:
            n_ensembles = n_ensembles_max
        return n_ensembles

    @staticmethod
    @set_random_state
    def assign(n_units: int, ensemble_size: int) -> Ensembles:
        """
        Assign units to ensembles by sub-sampling the units in distinct groups.

        Arguments
        ---------
        n_units : int
            See the argument :ref:`n_units` in the `process` method.
        ensemble_size : int
            See the configuration attribute `ensemble_size`.

        Returns
        -------
        ensembles : Ensembles
            See the returned value :ref:`ensembles` in the `process` method.

        Implementation
        --------------
        1. Shuffle units to distribute the members from each recording site.
        2. Split units into ensembles of size `ensemble_size`, except of the last ensemble which may
           include fewer units if the division is not exact.
           To determine the indices at which the array of units has to be split, start at
           `ensemble_size` and move by steps of `ensemble_size` up to the number of units.
           To split the array, use `np.split` with the array of units and the indices.
        3. Fill the last ensemble by randomly picking units from the previous ensembles, each one
           occurring at most once.

        See Also
        --------
        :func:`np.split(arr, indices)`
            Split an array into sub-arrays at the specified indices.
        :func:`np.random.choice(arr, size, replace)`
            Randomly pick elements from an array. Here, use `replace=False` to pick each element at
            most once.
        :func:`np.concatenate(arrays, axis)`
            Concatenate arrays along a given axis. If `axis` is None, then the arrays are flattened
            and concatenated end-to-end. Here, used twice: (1) to concatenate the units from the
            previous ensembles, (2) to concatenate the last ensemble with the picked units. Because
            the arrays are one-dimensional and the axis parameter is `None`, the arrays are joined
            end-to-end into a single 1D array containing all the elements in the order they appear
            in the input arrays.
        :func:`np.stack(arrays, axis)`
            Stack arrays along a new axis. Here, used to stack the ensembles into a 2D array. The
            axis parameter is set to 0 to stack the arrays along the first axis, such that the
            resulting array has the shape `(n_ensembles, ensemble_size)`.
        """
        units = np.arange(n_units)
        np.random.shuffle(units)
        # Split units into `q` full-sized ensembles of size `ensemble_size` and a last partial one
        split_indices = list(range(ensemble_size, n_units, ensemble_size))
        splits = np.split(units, split_indices)
        # Pick units from previous ensembles to complete the last one (if needed)
        n_missing = ensemble_size - splits[-1].size
        if n_missing > 0:
            candidate_units = np.concatenate(splits[:-1], axis=None)
            picked_units = np.random.choice(candidate_units, size=n_missing, replace=False)
            last_ensemble = np.concatenate((splits[-1], picked_units), axis=None)
            splits[-1] = last_ensemble
        # Stack ensembles
        ensembles = np.stack(splits, axis=0)
        return ensembles

    @staticmethod
    def limit_ensembles(ensembles: Ensembles, n_ensembles_max: int) -> Ensembles:
        """
        Keep only the first ensembles if the number of ensembles exceeds the maximum number.

        Arguments
        ---------
        ensembles : Ensembles
            Ensemble assignments. Shape: ``(n_ensembles, ensemble_size)``.
        n_ensembles_max : int
            Maximum number of ensembles to keep.

        Returns
        -------
        ensembles : Ensembles
            Ensemble assignments, limited to the maximum number of ensembles.
            Shape: ``(n_ensembles_max, ensemble_size)``.
        """
        if ensembles.shape[0] > n_ensembles_max:
            ensembles = ensembles[:n_ensembles_max]
        return ensembles
