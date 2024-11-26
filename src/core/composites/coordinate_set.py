#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.coordinate_set` [module]

Classes
-------
CoordinateSet
"""
from copy import deepcopy
from typing import List, Iterable, Tuple, Type, Self
import warnings

import numpy as np

from core.attributes.base_attribute import Attribute
from core.composites.attribute_set import AttributeSet, AttributeSetUnion
from core.coordinates.base_coord import Coordinate
from core.composites.base_container import Container


class CoordinateSet(Container[Type[Attribute], Coordinate]):
    """
    Implement coordinate-related logic for a collection of coordinates.

    This class behaves like a container for multiple coordinates, providing methods to operate on
    them consistently.

    Class Attributes
    ----------------
    KEY_TYPE : Type[Attribute]
        Type of the keys in the container, here types of attributes (classes).
    VALUE_TYPE : Coordinate
        Type of the values in the container, here coordinates instances.

    Attributes
    ----------
    See the base class `Container`.
    n_samples : int
        Number of samples common across the coordinates.

    Methods
    -------
    validate_length(*coords) -> None
    iter_through()
    filter_by_type(*attribute_types) -> Self
    match(query) -> np.ndarray
    match_single(query) -> np.ndarray
    match_union(query) -> np.ndarray
    match_idx(query) -> np.ndarray
    count(query) -> int
    """

    KEY_TYPE = type(Attribute)
    VALUE_TYPE = Coordinate

    def __repr__(self) -> str:
        attr_classes = sorted(self.keys(), key=lambda cls: cls.__name__)  # order by names
        return (
            f"CoordinateSet({', '.join(f'{self[cls]}' for cls in attr_classes)}, "
            f"#samples={self.n_samples})"
        )

    def __init__(self, *coords: Coordinate, **coords_dict: Coordinate) -> None:
        """
        Override the base constructor to fix `key_type` and `value_type` and to allow flexible
        input (iterables of coordinates).

        Arguments
        ---------
        *coords : Coordinate
            Coordinates to include in the set, passed as positional arguments.
        **coords_dict : Coordinate
            Coordinates to include in the set, passed as keyword arguments.

        Notes
        -----
        The names of the keyword arguments are not used. The presence of keyword arguments in the
        signature only aims to allow passing a dictionary of coordinates directly by unpacking,
        instead of requiring to extract the values.
        """
        # Validate all coordinates
        all_coords: List[Coordinate] = list(coords) + list(coords_dict.values())
        self.validate_length(*all_coords)
        # Retrieve their associated Attribute classes
        attr_classes = [coord.get_attribute() for coord in all_coords]
        # Create the dictionary of attributes with the class of the attribute as key
        data = {cls: coord for cls, coord in zip(attr_classes, all_coords)}
        # Pass data as positional argument to the parent class with fixed key and value types
        super().__init__(data, key_type=self.KEY_TYPE, value_type=self.VALUE_TYPE)

    @staticmethod
    def validate_length(*coords: Coordinate) -> None:
        """
        Ensures that all coordinates have the same number of samples.

        Raises
        ------
        ValueError
            If the number of samples is inconsistent across coordinates.
        """
        lengths = [len(coord) for coord in coords]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent coordinate lengths: {lengths}")

    def set(self, coord: Coordinate) -> None:
        """
        Set a new coordinate in the set, with a key corresponding to the class of the attribute.

        Arguments
        ---------
        coord : Coordinate
            Coordinate instance to set in the set.
        """
        if len(coord) != self.n_samples:  # validate length for the new coordinate
            raise ValueError(f"Invalid coordinate length: {len(coord)} != {self.n_samples}")
        key = coord.get_attribute()
        self[key] = coord

    def add(self, coord: Coordinate) -> Self:
        """
        Create a new instance of the set with an additional coordinate.

        Arguments
        ---------
        coord : Coordinate

        Returns
        -------
        self : CoordinateSet
            New instance with the added coordinate.
        """
        new_obj = deepcopy(self)  # deep copy to preserve the original set
        new_obj.set(coord)  # automatic validation
        return new_obj

    @property
    def n_samples(self) -> int:
        """Common number of samples across all coordinates."""
        return len(self[next(iter(self))])  # use the first coordinate

    def iter_through(self) -> Iterable[Tuple]:
        """
        Iterate over the samples across the coordinates, providing tuples which contain the joined
        values within the coordinates.

        Returns
        -------
        Iterable[Tuple]
            Iterable of tuples with the values of the coordinates, sample by sample.
        """
        return zip(*self.values())

    def filter_by_type(self, *attribute_types: Type[Attribute]) -> Self:
        """
        Filter coordinates associated with the specified attribute type(s).

        Arguments
        ---------
        attribute_types : Type[Attribute]
            Attribute type(s) to filter coordinates by.

        Returns
        -------
        filtered_coords : CoordinateSet
            Coordinates matching the specified attributes type or their subtypes, in a new
            `CoordinateSet` instance.

        See Also
        --------
        `Coordinate.has_attribute`
        """
        coords = [c for c in self.values() if any(c.has_attribute(tpe) for tpe in attribute_types)]
        return self.__class__(*coords)

    def match(self, query: AttributeSet | AttributeSetUnion) -> np.ndarray:
        """
        Generate a boolean mask to mark samples matching one or several attribute sets.

        Arguments
        ---------
        query : AttributeSet | AttributeSetUnion
            Attribute set or union to match against.

        Returns
        -------
        mask : np.ndarray
            Boolean mask indicating which samples match the values specified in the set(s).
        """
        if isinstance(query, AttributeSet):
            return self.match_single(query)
        elif isinstance(query, AttributeSetUnion):
            return self.match_union(query)
        else:
            raise ValueError(f"Invalid type for argument: {type(query)} not `AttributeSet`")

    def match_single(self, query: AttributeSet) -> np.ndarray:
        """
        Generate a boolean mask to mark samples matching a single set of attributes.

        Notes
        -----
        Rules if the attributes in the set do not match the attribute types of the coordinates:

        - If the attribute set is less specific than the coordinates (i.e. less factors than
          coordinates), then the coordinates which are not associated with any factor specified in
          the condition are ignored for matching.
        - If the attribute set is more specific than the coordinates (i.e. more factors than
          coordinates), then the factors which are not associated with any coordinate are ignored.

        This behavior is intended to allow for partial matching without raising errors and without
        requiring additional checks and manipulations for using the method.

        Examples
        --------
        >>> query = AttributeSet(Task('PTD'), Attention('a'), Category('R'))
        >>> task_coord = CoordTask(['PTD', 'CLK', 'PTD'])
        >>> attn_coord = CoordAttention(['a', 'p', 'a'])
        >>> categ_coord = CoordCategory(['R', 'T', 'R'])
        >>> coords = CoordinateSet(task_coord, attn_coord, categ_coord)
        >>> mask = coords.match(query)
        >>> mask
        array([ True, False,  True])

        See Also
        --------
        `AttributeSet.get`
            Get the value of an attribute type specified in the set. None if the attribute type is
            not present. This method is inherited from `UserDict`.
        """
        mask = np.ones(self.n_samples, dtype=bool)  # initialize the mask with all True values
        for coord in self.values():
            tpe = coord.get_attribute()  # attribute type associated with the coordinate
            attribute = query.get(tpe)  # value of the attribute type specified in the set
            if attribute is not None:
                mask &= coord == attribute  # apply the filter to the coordinate
            else:
                warnings.warn(f"Ignored coordinate: {coord.__class__}", UserWarning)
        return mask

    def match_union(self, x_union: AttributeSetUnion) -> np.ndarray:
        """
        Generate a boolean mask to index samples that match any of the sets in a union.

        Examples
        --------
        >>> x_1 = ExpCondition(Task('PTD'), Attention('a'), Category('R'))
        >>> x_2 = ExpCondition(Task('CLK'), Attention('a'), Category('T'))
        >>> union = x_1 + x_2
        >>> task_coord = CoordTask(['PTD', 'CLK', 'PTD'])
        >>> attn_coord = CoordAttention(['a', 'p', 'a'])
        >>> categ_coord = CoordCategory(['R', 'T', 'R'])
        >>> coords = CoordinateSet(task_coord, attn_coord, categ_coord)
        >>> mask = CoordinateSet(task_coord, attn_coord, categ_coord).match(union)
        >>> mask
        array([ True, False,  True])
        """
        all_masks = [self.match(query) for query in x_union]  # all masks for each set in the union
        mask = np.logical_or.reduce(all_masks)  # combine with a logical OR operation
        return mask

    def match_idx(self, query: AttributeSet | AttributeSetUnion) -> np.ndarray:
        """
        Find the indices of the samples that match the values in a set of attributes or union.

        Arguments
        ---------
        query : AttributeSet | AttributeSetUnion
            Attribute set or union to filter the coordinates by.

        Returns
        -------
        idx : np.ndarray[int]
            Indices of the samples that match the values in the set(s).
        """
        mask = self.match(query)
        return np.where(mask)[0]

    def count(self, query: AttributeSet | AttributeSetUnion) -> int:
        """
        Count the number of samples that match the values in a set of attributes or union.

        Arguments
        ---------
        query : AttributeSet | AttributeSetUnion
            Attribute set or union to match against.

        Returns
        -------
        n_samples : int
            Number of samples matching the values in the set(s).
        """
        mask = self.match(query)
        return np.sum(mask)
