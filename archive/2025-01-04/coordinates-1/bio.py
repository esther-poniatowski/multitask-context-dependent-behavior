#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.coordinates.bio` [module]

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

from coordinates.base_coord import Coordinate
from attributes.bio_info import CorticalDepth
from core.attributes.composites import Unit


T = TypeVar("T", Unit, CorticalDepth)
"""Generic type variable for neuronal features."""


class CoordPopulation(Coordinate, Generic[T]):
    """
    Coordinate labels representing features in a neuronal population : Units, CorticalDepth.

    Class Attributes
    ----------------
    feature: Type[T]
        Subclass of :class:`Entity` corresponding to the type of feature which is represented by
        the coordinate.

    Attributes
    ----------
    values: npt.NDArray[np.str_]
        Labels for the feature in one population. All the features are represented by string values,
        matching the values of the corresponding feature objects.

    See Also
    --------
    :class:`core.entities.composites.Unit`
    :class:`core.entities.bio_data.CorticalDepth`
    :class:`core.coordinates.base_coord.Coordinate`
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
        Units identifiers (attribute :attr:`id` of the unit objects).

    See Also
    --------
    :class:`core.entities.composites.Unit`
    :class:`core.coordinates.base_coord.CoordPopulation`
    """

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    @classmethod
    def build_labels(cls, units: Iterable[Unit]) -> npt.NDArray[np.str_]:
        """
        Build basic labels from a list of units.

        Parameters
        ----------
        units : Iterable[Unit]
            Units in the population.

        Returns
        -------
        values: npt.NDArray[np.str_]
            Labels corresponding to the identifier of each unit.

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
        Depth in the cortex (attribute :attr:`Unit.depth` of the unit objects).

    Methods
    -------
    :meth:`get_layer`
    :meth:`count_by_lab`

    See Also
    --------
    :class:`core.entities.bio_data.CorticalDepth`
    :class:`core.coordinates.coord_base.Coordinate`
    """

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    # pylint: disable=arguments-differ
    @classmethod
    def build_labels(cls, units: Iterable[Unit]) -> npt.NDArray[np.str_]:
        """
        Build basic labels from a list of units.

        Parameters
        ----------
        units : Iterable[Unit]
            Units in the population.

        Returns
        -------
        values: npt.NDArray[np.str_]
            Labels for the cortical depth for each unit.

        Raises
        ------
        TypeError
            If the input is not a list of units.
        """
        cls._check_population(units)
        return np.array([unit.depth.value for unit in units], dtype=np.str_)

    # pylint: enable=arguments-differ

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
