#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.features` [module]

Classes
-------
`Features`
"""

from typing import List, Iterable, Tuple, Type, Self
import warnings

import numpy as np

from core.entities.base_entity import Entity
from core.composites.exp_conditions import ExpCondition, ExpConditionUnion
from core.coordinates.base_coord import Coordinate


class Features:
    """
    Implement coordinate-related logic for a collection of coordinates.

    This class behaves like a container for multiple coordinates, providing methods to operate on
    them consistently.

    Class Attributes
    ----------------

    Attributes
    ----------
    coordinates : List[CoordExpFactor]
        List of coordinates to manage.
    register : List[str]
        Names of the coordinates, if provided.
    n_samples : int
        Number of samples common across the coordinates.

    Methods
    -------
    `validate_length`
    `__len__`
    `__repr__`
    `__iter__`
    `filter_by_type`
    `match`
    `match_single`
    `match_union`
    `match_idx`
    `count`
    """

    def __init__(self, *coords: Coordinate, **coords_dict: Coordinate) -> None:
        # Register the names of the coordinates if provided
        self.register: List[str] = ["" for _ in range(len(coords))] + list(coords_dict.keys())
        # Add the coordinates from the dictionary to the list of coordinates
        self.coords: List[Coordinate] = list(coords) + list(coords_dict.values())
        # Validate that all coordinates have the same length
        self.validate_length(*self.coords)

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

    @property
    def n_samples(self) -> int:
        """Common number of samples across all coordinates."""
        return len(self.coords[0])

    def __len__(self) -> int:
        """Number of coordinates managed."""
        return len(self.coords)

    def __repr__(self) -> str:
        coord_types = [f"{type(coord).__name__}" for coord in self.coords]
        return f"CoordinateSet({', '.join(coord_types)}), #samples: {self.n_samples}"

    def __iter__(self) -> Iterable[Tuple]:
        """
        Iterate over the coordinates, providing tuples of elements which contain the joined
        values within the coordinates, sample by sample.

        Returns
        -------
        Iterable[Tuple]
            Iterable of tuples with the values of the coordinates for each sample.
        """
        return zip(*self.coords)

    def filter_by_type(self, *entity_types: Type[Entity]) -> Self:
        """
        Filter coordinates associated with the specified entity type(s).

        Arguments
        ---------
        entity_types : Type[Entity]
            Entity type(s) to filter coordinates by.

        Returns
        -------
        filtered_coords : Features
            Coordinates matching the specified entities type or their subtypes, in a new
            `Features` instance.

        See Also
        --------
        `Coordinate.has_entity`
        """
        retained = [c for c in self.coords if any(c.has_entity(tpe) for tpe in entity_types)]
        return self.__class__(*retained)

    def match(self, exp_cond: ExpCondition | ExpConditionUnion) -> np.ndarray:
        """
        Generate a boolean mask to mark samples matching one or several experimental condition.

        Arguments
        ---------
        exp_cond : ExpCondition | ExpConditionUnion
            Experimental condition or union of conditions to match against.

        Returns
        -------
        mask : np.ndarray
            Boolean mask indicating which samples match the condition(s).
        """
        if isinstance(exp_cond, ExpCondition):
            return self.match_single(exp_cond)
        elif isinstance(exp_cond, ExpConditionUnion):
            return self.match_union(exp_cond)
        else:
            raise ValueError(f"Invalid type for exp_cond: {type(exp_cond)}")

    def match_single(self, exp_cond: ExpCondition) -> np.ndarray:
        """
        Generate a boolean mask to mark samples matching a single experimental condition.

        Notes
        -----
        Rules if the entities in the condition does not match the entities in the coordinates:

        - If the condition is less specific than the coordinates (i.e. less factors than
          coordinates), then the coordinates which are not associated with any factor specified in
          the condition are ignored for matching.
        - If the condition is more specific than the coordinates (i.e. more factors than
          coordinates), then the factors which are not associated with any coordinate are ignored.

        This behavior is intended to allow for partial matching of conditions and coordinates
        without raising errors and without requiring additional checks and manipulations for using
        the method.

        Examples
        --------
        >>> task = Task('PTD')
        >>> attention = Attention('a')
        >>> category = Category('R')
        >>> exp_cond = ExpCondition(task=task, attention=attention, category=category)
        >>> task_coord = CoordTask(np.array(['PTD', 'CLK', 'PTD']))
        >>> attn_coord = CoordAttention(np.array(['a', 'p', 'a']))
        >>> categ_coord = CoordCategory(np.array(['R', 'T', 'R']))
        >>> coords = Features(task_coord, attn_coord, categ_coord)
        >>> mask = coords.match(exp_cond)
        >>> mask
        array([ True, False,  True])

        See Also
        --------
        `ExpCondition.get_factor`
            Get the value of a factor specified in the condition. None if the factor is not present.
        """
        # Initialize the mask with all True values
        mask = np.ones(self.n_samples, dtype=bool)
        for coord in self.coords:
            # Retrieve the entity associated with the coordinate
            entity = coord.get_entity()
            # Get the value of the factor specified in the condition
            factor = exp_cond.get_factor(entity)
            if factor is not None:
                # Apply the condition to the coordinate
                mask &= coord == factor
            else:
                warnings.warn(f"Ignored coordinate: {coord.__class__}", UserWarning)
        return mask

    def match_union(self, exp_cond: ExpConditionUnion) -> np.ndarray:
        """
        Generate a boolean mask to index samples that match any of the conditions in the union.

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task=Task('PTD'), attention=Attention('a'), category=Category('R'))
        >>> exp_cond_2 = ExpCondition(task=Task('CLK'), attention=Attention('a'), category=Category('T'))
        >>> union = exp_cond_1 + exp_cond_2
        >>> task_coord = CoordTask(np.array(['PTD', 'CLK', 'PTD']))
        >>> attn_coord = CoordAttention(np.array(['a', 'p', 'a']))
        >>> categ_coord = CoordCategory(np.array(['R', 'T', 'R']))
        >>> coords = Features(task_coord, attn_coord, categ_coord)
        >>> mask = Features(task_coord, attn_coord, categ_coord).match(union)
        >>> mask
        array([ True, False,  True])
        """
        # Get the masks for each condition in the union using the method for a single condition
        all_masks = [self.match(cond) for cond in exp_cond]
        # Combine all masks with a logical OR operation
        mask = np.logical_or.reduce(all_masks)
        return mask

    def match_idx(self, exp_cond: ExpCondition | ExpConditionUnion) -> np.ndarray:
        """
        Find the indices of the samples that match the condition(s).

        Arguments
        ---------
        exp_cond : ExpCondition, ExpConditionUnion
            Experimental condition or union of conditions to filter the coordinates by.

        Returns
        -------
        idx : np.ndarray[int]
            Indices of the samples that match the condition(s).
        """
        mask = self.match(exp_cond)
        return np.where(mask)[0]

    def count(self, exp_cond: ExpCondition | ExpConditionUnion) -> int:
        """
        Count the number of samples that match an experimental condition.

        Arguments
        ---------
        exp_cond : ExpCondition | ExpConditionUnion
            Experimental condition or union of conditions to match against.

        Returns
        -------
        n_samples : int
            Number of samples matching the condition.
        """
        mask = self.match(exp_cond)
        return np.sum(mask)
