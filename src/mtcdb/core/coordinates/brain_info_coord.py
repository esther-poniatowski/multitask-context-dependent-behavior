"""
`core.coordinates.bio` [module]

Coordinate for labelling biological information in neuronal populations.

Classes
-------
CoordPopulation (Generic)
CoordUnit
CoordDepth
"""
from typing import Dict, Tuple, Any, Iterable

import numpy as np

from core.coordinates.base_coordinate import Coordinate
from core.attributes.brain_info import CorticalDepth
from core.attributes.brain_info import Unit


class CoordUnit(Coordinate[Unit]):
    """
    Coordinate labels for the units (neurons) in a population.

    Class Attributes
    ----------------
    ATTRIBUTE : Type[Unit]
        Subclass of `Attribute` corresponding to the units identifiers.
    DTYPE : Type[np.str_]
        Data type of the unit labels, always string.
    SENTINEL : str
        Sentinel value marking missing or unset unit labels, usually an empty string.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.str_]
        Labels for the units in one population.
        Shape: ``(n_units,)`` for a single ensemble, ``(n_ens, n_units)`` for multiple ensembles.

    Methods
    -------
    iter_through

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
    `core.attributes.brain_info.Unit`
    `core.coordinates.base_coordinate.Coordinate`
    """

    ATTRIBUTE = Unit
    DTYPE = np.dtype("str")
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


class CoordDepth(Coordinate[CorticalDepth]):
    """
    Coordinate indicating the cortical depth of each unit in a population.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.str_]
        Depth of units in the cortex.

    Methods
    -------
    get_layer
    count_by_lab

    See Also
    --------
    `core.attributes.bio_data.CorticalDepth`
    `core.coordinates.coord_base.Coordinate`
    """

    ATTRIBUTE = CorticalDepth
    DTYPE = np.dtype("str")
    SENTINEL = ""

    def __repr__(self) -> str:
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{depth!r}: {n}" for depth, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} units, {format_counts}."

    def get_layer(self, depth: CorticalDepth) -> np.ndarray[Tuple[Any, ...], np.dtype[np.bool_]]:
        """
        Identify the units belonging to one layer (i.e depth).

        Parameters
        ----------
        depth : CorticalDepth
            Layer to select. Behaves like a string.

        Returns
        -------
        mask : np.ndarray[np.bool_]
            Boolean mask for the units in the layer.
            Shape : ``(n_smpl,)``, same as the coordinate.

        See Also
        --------
        `numpy.equal`: Used to compare the coordinate values to the depth value, instead of the `==`
        operator (which raises type errors due to inheritance of `np.ndarray` for the coordinate and
        inheritance of `str` for the depth).
        """
        return np.equal(self, depth)

    def count_by_lab(self) -> Dict[CorticalDepth, int]:
        """
        Count the number of units in each layer.

        Returns
        -------
        counts : Dict[CorticalDepth, int]
            Number of units in each layer.
        """
        options = self.ATTRIBUTE.get_options()
        return {self.ATTRIBUTE(depth): np.sum(self == depth) for depth in options}
