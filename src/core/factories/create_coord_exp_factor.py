"""
`core.factories.create_coord_exp_factor` [module]

Classes
-------
FactoryCoordExpFactor
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `create` method in `FactoryCoordExpFactor`
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------

from typing import Type, Dict, Iterable, Tuple
from functools import cached_property

from core.factories.base_factory import Factory
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.composites.exp_conditions import ExpCondition


class FactoryCoordExpFactor(Factory[CoordExpFactor]):
    """
    Create coordinates for experimental factors.

    Product: `CoordExpFactor`

    Attributes
    ----------
    counts_by_condition : Dict[ExpCondition, int]
        Number of trials for each experimental condition of interest.
    order_conditions : Iterable[ExpCondition]
        Order in which the experimental conditions appear in the output coordinates.
    n_trials : int
        (Property) Total number of trials in the output coordinates.
    conditions_boundaries : Dict[ExpCondition, Tuple[int, int]]
        (Property) Start and end indices of the trials for each condition along the trials dimension
        in the output coordinates.

    Methods
    -------
    create (required)
    validate_factor

    Examples
    --------
    Set the number of trials for each experimental condition:

    >>> exp_condition_1 = ExpCondition(task='PTD', attention='a')
    >>> exp_condition_2 = ExpCondition(task='CLK', attention='a')
    >>> counts_by_condition = {exp_condition_1: 3, exp_condition_2: 2}
    >>> order_conditions = (exp_condition_1, exp_condition_2)
    >>> factory = FactoryCoordExpFactor(counts_by_condition, order_conditions)
    >>> factory.conditions_boundaries
    {ExpCondition(task='PTD', attention='a'): (0, 3),
     ExpCondition(task='CLK', attention='a'): (3, 5)}

    Create the task coordinate for the pseudo-trials:

    >>> factory.create(coord_type=CoordTask)
    CoordTask(['PTD', 'PTD', 'PTD', 'CLK', 'CLK'])

    """

    PRODUCT_CLASSES = CoordExpFactor

    def __init__(
        self,
        counts_by_condition: Dict[ExpCondition, int],
        order_conditions: Iterable[ExpCondition],
    ) -> None:
        # Check identical conditions in both attributes
        if set(counts_by_condition.keys()) != set(order_conditions):
            raise ValueError("Different conditions in `counts_by_condition` and `order_conditions`")
        # Store configuration parameters
        self.counts_by_condition = counts_by_condition
        self.order_conditions = order_conditions

    def create(self, coord_type: Type[CoordExpFactor]) -> CoordExpFactor:
        """
        Implement the base class method.

        Parameters
        ----------
        coord_type : Type[CoordExpFactor]
            Class of the coordinate to create, representing one experimental factor specified in the
            all the conditions of the `counts_by_condition` attribute. .. _coord_type:

        Returns
        -------
        coord : CoordExpFactor
            Coordinates for the experimental factor in the pseudo-trials.
        """
        self.validate_factor(coord_type)
        # Initialize empty coordinates with as many trials as the total number of pseudo-trials
        coord = coord_type.from_shape(self.n_trials)
        # Fill the coordinates by condition
        attribute_type = coord_type.get_attribute()  # attribute associated with the coordinate
        for cond, (start, end) in self.conditions_boundaries.items():
            value = cond.get(attribute_type)  # value to fill
            coord[start:end] = value
        return coord

    @cached_property
    def n_trials(self) -> int:
        """Get the total number of trials in the output coordinates."""
        return sum(self.counts_by_condition.values())

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
            n_trials = self.counts_by_condition[cond]
            end = start + n_trials
            idx_start_end[cond] = (start, end)
            start = end
        return idx_start_end

    def validate_factor(self, coord_type: Type[CoordExpFactor]) -> None:
        """
        Check that the type of the coordinate to create is relevant for the trial dimension in the
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
            If the experimental factor is absent from one experimental condition of interest.
        """
        attribute_type = coord_type.get_attribute()  # attribute associated with the coordinate
        for cond in self.order_conditions:
            if attribute_type not in cond:  # behavior like a dictionary
                raise TypeError(
                    f"Experimental factor of {coord_type} not present in condition {cond}."
                )
