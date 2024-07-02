#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates.bio` [module]

Coordinate for labelling biological information in neuronal populations.

Classes
-------
:class:`CoordPopulation` (Generic)
:class:`CoordUnit`
:class:`CoordDepth`
"""
from typing import TypeVar, Type, Generic, Dict, List

import numpy as np
import numpy.typing as npt

from mtcdb.coordinates.base_coord import Coordinate
from mtcdb.core_objects.bio import CorticalDepth
from mtcdb.core_objects.composites import Unit



T = TypeVar('T', Unit, CorticalDepth)
"""Generic type variable for neuronal features."""


class CoordPopulation(Coordinate, Generic[T]):
    """
    Classes representing features in a neuronal population : 
    Units, CorticalDepth.

    Class Attributes
    ----------------
    feature: Type[T]
        Reference to the class of the neuronal feature object.

    Attributes
    ----------
    values: npt.NDArray[np.str_]
        Feature labels for the population.
        It contains the string values of the features.

    See Also
    --------
    :class:`mtcdb.core_objects.composites.Unit`
    :class:`mtcdb.core_objects.bio.CorticalDepth`
    :class:`mtcdb.coordinates.base_coord.Coordinate`
    """
    feature: Type[T]

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    @classmethod
    def build_labels(cls, units: List[Unit]) -> npt.NDArray[np.str_]:
        """
        Build coordinate labels from a list of units.

        Parameters
        ----------
        units : List[Unit]
            List of units in the population.
        
        Returns
        -------
        values: npt.NDArray[np.str_]
            Coordinate containing the appropriate label for each unit.
        """
        if cls.feature == CorticalDepth:
            return np.array([unit.depth.value for unit in units], dtype=np.str_)
        elif cls.feature == Unit:
            return np.array([unit.id for unit in units], dtype=np.str_)
        else:
            raise ValueError("Unknown feature type.")


class CoordUnit(CoordPopulation[Unit]):
    """
    Coordinate labels for the units (neurons) in a population.

    Attributes
    ----------
    values: npt.NDArray[np.str_]
        Units identifiers (attribute ``id`` of the unit objects).

    See Also
    --------
    :class:`mtcdb.core_objects.composites.Unit`
    :class:`mtcdb.coordinates.base_coord.CoordPopulation`
    """
    feature = Unit

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)


class CoordDepth(CoordPopulation[CorticalDepth]):
    """
    Coordinate labels for the depth of each unit in the brain.

    Attributes
    ----------
    values: npt.NDArray[np.str_]
        Depth in the cortex (attribute ``depth`` of the unit objects).

    Methods
    -------
    get_layer
    count_by_lab

    See Also
    --------
    :class:`mtcdb.core_objects.bio.CorticalDepth`
    :class:`mtcdb.coordinates.coord_base.Coordinate`
    """
    feature = CorticalDepth

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    def get_layer(self, depth: CorticalDepth) -> npt.NDArray[np.bool_]:
        """
        Identify the units belonging to one layer (i.e depth).

        Parameters
        ----------
        depth: CorticalDepth
            Layer to select.
        
        Returns
        -------
        mask: npt.NDArray[np.bool_]
            Boolean mask for the units in the layer.
            Shape : ``(n_smpl,)``
        """
        return self.values == depth.value

    def count_by_lab(self) -> Dict[CorticalDepth, int]:
        """
        Count the number of units in each layer.
        
        Returns
        -------
        n_u: Dict[CorticalDepth, int]
            Number of units in each layer.
        """
        return {depth: np.sum(self.values == depth.value) for depth in CorticalDepth.get_options()}
