#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_trial_coords` [module]

Classes
-------
`TrialCoordsBuilder`
"""
from typing import Type, Dict, Iterable, Tuple
from functools import cached_property

from core.builders.base_builder import CoordinateBuilder
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.composites.exp_conditions import ExpCondition


class TrialCoordsBuilder(CoordinateBuilder[CoordExpFactor]):
    """
    Build coordinates for experimental factors.

    Product: `CoordExpFactor`

    Attributes
    ----------
    counts_by_condition : Dict[ExpCondition, int]
        Number of trials for each experimental condition of interest.
    order_conditions : Iterable[ExpCondition]
        Order in which the experimental conditions appear in the output coordinates.
    conditions_boundaries : Dict[ExpCondition, Tuple[int, int]]
        Start and end indices of the trials for each condition along the trials dimension in the
        output coordinates.

    Methods
    -------
    `build` (required)
    `validate_factor`
    `construct_coord`

    Examples
    --------
    Set the number of trials for each experimental condition:

    >>> exp_cond_1 = ExpCondition(task='PTD', attention='a')
    >>> exp_cond_2 = ExpCondition(task='CLK', attention='a')
    >>> counts_by_condition = {exp_cond_1: 3, exp_cond_2: 2}
    >>> order_conditions = (exp_cond_1, exp_cond_2)
    >>> builder = TrialCoordsBuilder(counts_by_condition, order_conditions)
    >>> builder.conditions_boundaries
    {ExpCondition(task='PTD', attention='a'): (0, 3),
     ExpCondition(task='CLK', attention='a'): (3, 5)}

    Build the task coordinate for the pseudo-trials:

    >>> builder.build(coord_type=CoordTask)
    CoordTask(['PTD', 'PTD', 'PTD', 'CLK', 'CLK'])

    """

    PRODUCT_CLASS = CoordExpFactor

    def __init__(
        self, counts_by_condition: Dict[ExpCondition, int], order_conditions: Iterable[ExpCondition]
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Check identical conditions in both attributes
        if set(counts_by_condition.keys()) != set(order_conditions):
            raise ValueError("Different conditions in `counts_by_condition` and `order_conditions`")
        # Store configuration parameters
        self.counts_by_condition = counts_by_condition
        self.order_conditions = order_conditions

    def build(self, coord_type: Type[CoordExpFactor] | None = None, **kwargs) -> CoordExpFactor:
        """
        Implement the base class method.

        Parameters
        ----------
        coord_type : Type[CoordExpFactor]
            Class of the coordinate to build, representing one experimental factor specified in the
            all the conditions of the `counts_by_condition` attribute. .. _coord_type:

        Returns
        -------
        """
        assert coord_type is not None
        self.validate_factor(coord_type)
        self.product = self.construct_coord(coord_type)
        return self.get_product()

    def validate_factor(self, coord_type: Type[CoordExpFactor]) -> None:
        """
        Check that the type of the coordinate to build is relevant for the trial dimension in the
        current pipeline.

        Rule: The coordinate type is valid if the experimental factor associated with the coordinate
        is present in all the experimental conditions specified in the `counts_by_condition`
        attribute. Otherwise, this coordinate would not find any value for the trials belonging to
        the experimental condition in which the factor is missing.

        Parameters
        ----------
        coord_type : ExpFactor
            See the attribute `coord_type`.

        Raises
        ------
        TypeError
            If the experimental factor is not present in all the conditions.
        """
        for cond in self.order_conditions:
            for attribute_type in cond.get_attributes():
                if not coord_type.has_attribute(attribute_type):
                    raise TypeError(
                        f"Experimental factor of {coord_type} not present in condition {cond}."
                    )

    @cached_property
    def conditions_boundaries(self) -> Dict[ExpCondition, Tuple[int, int]]:
        """
        Set the start and end indices of the trials of each condition in the final coordinates.

        Those indices are used to ensure the consistency of the data between all the coordinates for
        the trials dimension.

        Returns
        -------
        idx_start_end : Dict[ExpCondition, Tuple[int, int]]
        """
        idx_start_end: Dict[ExpCondition, Tuple[int, int]] = {}
        start = 0
        for cond in self.order_conditions:
            n_samples = self.counts_by_condition[cond]
            end = start + n_samples
            idx_start_end[cond] = (start, end)
            start = end
        return idx_start_end

    def construct_coord(
        self,
        coord_type: Type[CoordExpFactor],
    ) -> CoordExpFactor:
        """
        Construct the trial coordinates for the pseudo-trials.

        Arguments
        ---------
        coord_type : Type[CoordExpFactor]
            Class of the coordinate to build.

        Returns
        -------
        coord : CoordExpFactor
            Coordinate for the trial dimension in the data structure product.
        """
        # Initialize empty coordinates with as many trials as the total number of pseudo-trials
        n_samples_tot = sum(self.counts_by_condition.values())
        coord = coord_type.from_shape(n_samples_tot)
        # Fill the coordinates by condition
        for cond, (start, end) in self.conditions_boundaries.items():
            # Retrieve the value in the exp condition for the coordinate attribute
            value = cond.get_factor(coord_type.get_attribute())
            coord[start:end] = value
        return coord
