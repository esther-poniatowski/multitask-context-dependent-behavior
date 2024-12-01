#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.factories.create_ensembles` [module]

Classes
-------
EnsemblesBuilder
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `build` method in `EnsemblesBuilder`
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------

from typing import List

import numpy as np

from core.builders.base_builder import Builder
from core.coordinates.brain_info_coord import CoordUnit
from core.processors.preprocess.assign_ensembles import EnsembleAssigner, Ensembles
from core.attributes.brain_info import Unit


class EnsemblesBuilder(Builder[CoordUnit]):
    """
    Build ensembles of units to form pseudo-populations for cross-validation.

    Product: `CoordUnits`

    Attributes
    ----------
    ensemble_size : int
        Number of units required to form each ensemble (imposed by the area with the lowest number
        of units).
    n_ensembles_max : int
        Maximum number of ensembles to form from the population (to limit the number of ensembles
        and computational cost for the areas with the largest populations).

    Methods
    -------
    build (required)
    construct_coord

    Examples
    --------
    Set the number of units in each ensemble (size) and the maximum number of ensembles:

    >>> builder = EnsemblesBuilder(ensemble_size=20, n_ensembles_max=10)

    Generate ensembles for a population of units:

    >>> builder.build(units=units, seed=0)

    """

    PRODUCT_CLASS = CoordUnit

    def __init__(self, ensemble_size: int, n_ensembles_max: int) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters
        self.ensemble_size = ensemble_size
        self.n_ensembles_max = n_ensembles_max

    def build(self, units: List[Unit], seed: int = 0) -> CoordUnit:
        """
        Implement the base class method.

        Arguments
        ---------
        units : List[Unit]
            Units in the population. Each element behaves like a string, representing the unit's
            identifier.
        seed : int
            Seed for the random number generator, used in the ensemble assignment.

        Returns
        -------
        ensembles : Ensembles
            Indices of the units in each ensemble. Shape: ``(n_ensembles, ensemble_size)``.
        """
        if units is None:
            units = []
        n_units = len(units)
        # Generate ensembles of units to form the pseudo-population
        assigner = EnsembleAssigner(self.ensemble_size, self.n_ensembles_max)
        ensembles = assigner.process(n_units=n_units, seed=seed)
        # Construct the units coordinate
        self.product = self.construct_coord(units, ensembles)
        return self.get_product()

    def construct_coord(self, units: List[Unit], ensembles: Ensembles) -> CoordUnit:
        """
        Construct the units coordinate for the pseudo-population (unit labels in each ensemble).

        Returns
        -------
        units : CoordUnit
            See the attribute `units` in the data structure product.

        Warning
        -------
        Two indices are used to identify the units in the population:

        - ``u_pop``: index of the unit in the population list (from 0 to ``n_units``).
        - ``u_ens``: index of the unit in the ensemble (from 0 to ``ensemble_size``).

        Implementation
        --------------
        Advanced indexing and broadcasting:

        ``coord[:, :] = np.array(units)[ensembles]``

        - Select elements from the units array using the indices specified in ensembles.
        - Broadcast the units array to the shape of ensembles.

        See Also
        --------
        `Coordinate.from_shape`
        """
        n_ensembles, _ = ensembles.shape
        coord = CoordUnit.from_shape((n_ensembles, self.ensemble_size))
        coord[:, :] = np.array(units)[ensembles]
        return coord
