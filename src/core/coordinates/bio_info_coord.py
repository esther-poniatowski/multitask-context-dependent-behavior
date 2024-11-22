#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.bio` [module]

Coordinate for labelling biological information in neuronal populations.

Classes
-------
`CoordPopulation` (Generic)
`CoordUnit`
`CoordDepth`
"""
from typing import Dict, Tuple, Any, Iterable

import numpy as np

from core.coordinates.base_coord import Coordinate
from core.entities.bio_info import CorticalDepth
from core.entities.bio_info import Unit


class CoordUnit(Coordinate[np.str_, Unit]):
    """
    Coordinate labels for the units (neurons) in a population.

    Class Attributes
    ----------------
    ENTITY : Type[Unit]
        Subclass of `Entity` corresponding to the units identifiers.
    DTYPE : Type[np.str_]
        Data type of the unit labels, always string.
    SENTINEL : str
        Sentinel value marking missing or unset unit labels, usually an empty string.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any, ...], np.str_]
        Labels for the units in one population.

        Dimensions:

        - Single ensemble: ``(n_units,)``.
        - Multiple ensembles: ``(n_ens, n_units)``.

    Examples
    --------
    Create a coordinate for 3 units forming a single ensemble:

    >>> units = ['avo052a-d1', 'avo052a-d2', 'lemon024b-d3']
    >>> coord = CoordUnit(units)
    >>> coord
    <CoordUnit>: 3 units.

    Create a coordinate for 2 ensembles, each with 3 units:

    >>> units = [['avo052a-d1', 'avo052a-d2', 'lemon024b-d3'],
    ...          ['avo052a-d1', 'avo052a-d2', 'lemon024b-d3']]
    >>> coord = CoordUnit(units)
    >>> coord
    <CoordUnit>: 2 ensembles, 3 units each.

    See Also
    --------
    `core.entities.bio_info.Unit`
    `core.coordinates.base_coord.Coordinate`
    """

    ENTITY = Unit
    DTYPE = np.str_
    SENTINEL = ""

    def __repr__(self) -> str:
        if self.ndim == 1:
            return f"<{self.__class__.__name__}>: {len(self)} units."
        n_ens, n_units = self.shape
        return f"<{self.__class__.__name__}>: {n_ens} ensembles, {n_units} units each."

    def iter_through(self, iterable: Iterable[Any], units: Iterable[Unit]) -> Iterable:
        """
        Iterate over an iterable of values associated to units in the initial population, and yield
        the values of the units selected in the ensemble.

        Arguments
        ---------
        iterable : Iterable
            Iterable of values associated to the units in the population.
        units : Iterable[Unit]
            Units in the population (identifiers as in the coordinate).
        Yields
        ------
        value : Any
            Value associated to the unit in the ensemble.

        Examples
        --------
        Iterate over the firing rates of the units in an ensemble specified by a coordinate:

        >>> coord_units = CoordUnit(['avo052a-d1', 'avo052a-d2'])
        >>> units = ['avo052a-d1', 'lemon023a-b1', 'avo052a-d2', 'lemon024b-d3']
        >>> firing_rates = [0.1, 0.2, 0.3, 0.4]
        >>> for fr in coord_units.iter_through(firing_rates, units):
        ...    print(fr)
        0.1
        0.3

        Notes
        -----
        For generality, iteration is performed via a dictionary of unit-value pairs rather than via
        the `index` method and the bracket indexing of the iterable (which are not necessarily
        supported by all iterable types).
        """
        unit_value_dict = dict(zip(units, iterable))
        for unit in self:
            yield unit_value_dict[unit]


class CoordDepth(Coordinate[np.str_, CorticalDepth]):
    """
    Coordinate indicating the cortical depth of each unit in a population.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any, ...], np.str_]
        Depth of units in the cortex.

    Methods
    -------
    `get_layer`
    `count_by_lab`

    See Also
    --------
    `core.entities.bio_data.CorticalDepth`
    `core.coordinates.coord_base.Coordinate`
    """

    ENTITY = CorticalDepth
    DTYPE = np.str_
    SENTINEL = ""

    def get_layer(self, depth: CorticalDepth) -> np.ndarray[Tuple[Any, ...], np.dtype[np.bool_]]:
        """
        Identify the units belonging to one layer (i.e depth).

        Parameters
        ----------
        depth : CorticalDepth
            Layer to select.

        Returns
        -------
        mask : np.ndarray[np.bool_]
            Boolean mask for the units in the layer.
            Shape : ``(n_smpl,)``, same as the coordinate.
        """
        return self == depth.value

    def count_by_lab(self) -> Dict[CorticalDepth, int]:
        """
        Count the number of units in each layer.

        Returns
        -------
        counts : Dict[CorticalDepth, int]
            Number of units in each layer.
        """
        options = self.ENTITY.get_options()
        return {self.ENTITY(depth): np.sum(self == depth) for depth in options}

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{depth!r}: {n}" for depth, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} units, {format_counts}."
