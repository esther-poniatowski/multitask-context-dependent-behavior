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
from typing import TypeVar, Generic, Dict, Iterable, Any

import numpy as np
import numpy.typing as npt

from mtcdb.coordinates.base import Coordinate
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
    :class:`mtcdb.coordinates.base.Coordinate`
    """
    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    @staticmethod
    def _check_population(pop: Iterable[Any]) -> None:
        """
        Check if the input is a list of units (neurons).

        Parameters
        ----------
        pop : List[Any]
            List of objects to check.
        
        Raises
        ------
        TypeError
            If the input is not a list of the feature objects.
        """
        if not all(isinstance(obj, Unit) for obj in pop):
            raise TypeError("Invalid Argument")


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
    :class:`mtcdb.coordinates.base.CoordPopulation`
    """
    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    @classmethod
    def build_labels(cls, units: Iterable[Unit]) -> npt.NDArray[np.str_]: # pylint: disable=arguments-differ
        """
        Build coordinate labels from a list of units.

        Parameters
        ----------
        units : Iterable[Unit]
            Units in the population.
        
        Returns
        -------
        values: npt.NDArray[np.str_]
            Coordinate for the id for each unit.
        
        Raises
        ------
        TypeError
            If the input is not a list of units.
        """
        cls._check_population(units)
        return np.array([unit.id for unit in units], dtype=np.str_)


class CoordDepth(CoordPopulation[CorticalDepth]):
    """
    Coordinate indicating the cortical depth of each unit in a population.

    Attributes
    ----------
    values: npt.NDArray[np.str_]
        Depth in the cortex (attribute ``depth`` of the unit objects).

    Methods
    -------
    :meth:`get_layer`
    :meth:`count_by_lab`

    See Also
    --------
    :class:`mtcdb.core_objects.bio.CorticalDepth`
    :class:`mtcdb.coordinates.coord_base.Coordinate`
    """
    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    @classmethod
    def build_labels(cls, units: Iterable[Unit]) -> npt.NDArray[np.str_]: # pylint: disable=arguments-differ
        """
        Build coordinate labels from a list of units.

        Parameters
        ----------
        units : Iterable[Unit]
            Units in the population.
        
        Returns
        -------
        values: npt.NDArray[np.str_]
            Coordinate for the cortical depth for each unit.

        Raises
        ------
        TypeError
            If the input is not a list of units.
        """
        cls._check_population(units)
        return np.array([unit.depth.value for unit in units], dtype=np.str_)

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
        counts: Dict[CorticalDepth, int]
            Number of units in each layer.
        """
        options = CorticalDepth.get_options()
        return {CorticalDepth(depth): np.sum(self.values == depth) for depth in options}
