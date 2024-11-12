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
from typing import Dict, Tuple, Any

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
        Labels for the units in one population. One or several dimensions, for single and multiple
        ensemble respectively.

    See Also
    --------
    :class:`core.entities.composites.Unit`
    :class:`core.coordinates.base_coord.Coordinate`
    """

    ENTITY = Unit
    DTYPE = np.str_
    SENTINEL = ""


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
    :class:`core.entities.bio_data.CorticalDepth`
    :class:`core.coordinates.coord_base.Coordinate`
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
