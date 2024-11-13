#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.exp_conditions` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
`ExpCondition`
"""

from typing import Self, Iterable, List, Sequence
from itertools import product

import numpy as np

from core.entities.exp_factors import (
    Task,
    Attention,
    Category,
    Behavior,
    ResponseOutcome,
    ExpFactor,
)
from core.coordinates.exp_factor_coord import (
    Coordinate,
    CoordTask,
    CoordAttention,
    CoordCategory,
    CoordBehavior,
    CoordOutcome,
)


class ExpCondition:
    """
    Experimental condition defined by the combination of several experimental factors, among:

    - task
    - attentional state
    - category of the stimulus
    - behavioral choice

    Class Attributes
    ----------------
    FACTOR_TO_ATTR : Dict[ExpFactor, str]
        Mapping between the experimental factor classes and the corresponding attribute names.
    COORD_TO_ATTR : Dict[Coordinate, str]
        Mapping between the coordinate classes and the corresponding attribute names.

    Attributes
    ----------
    task : Task
        Task performed by the animal.
    attention : Attention
        Attentional state in which the animal is engaged.
    category : Category
        Category of the stimulus presented to the animal.
    behavior : Behavior
        Behavioral choice of the animal (defined in the attentive state).
    outcome : ResponseOutcome
        Outcome of the behavioral choice (defined in the attentive state).

    Methods
    -------
    `combine_factors`
    `__add__`
    `union`
    `match`
    `count`

    Notes
    -----
    Partial conditions can be defined by specifying only the factors of interest. The remaining
    factors are set to `None` by default.

    Warning
    -------
    Instantiate with keyword arguments to avoid confusion between the factors, especially when
    partial conditions are defined.

    Examples
    --------
    Initialize a partial condition with three factors (without specifying the 'attention' factor):

    >>> task = Task('PTD')
    >>> category = Category('R')
    >>> behavior = Behavior('Go')
    >>> exp_cond_1 = ExpCondition(task=task, category=category, behavior=behavior)
    >>> exp_cond_1
    ExpCondition(task=PTD, category=R, behavior=Go)

    Initialize directly with strings:

    >>> exp_cond_2 = ExpCondition(task='CLK', attention='p', category='T')

    Retrieve the task:

    >>> exp_cond_1.task
    Task('PTD')

    See Also
    --------
    `Task`
    `Attention`
    `Category`
    `Behavior`
    `ResponseOutcome`
    `ExpConditionUnion`
    """

    FACTOR_TO_ATTR = {
        Task: "task",
        Attention: "attention",
        Category: "category",
        Behavior: "behavior",
        ResponseOutcome: "outcome",
    }
    COORD_TO_ATTR = {
        CoordTask: "task",
        CoordAttention: "attention",
        CoordCategory: "category",
        CoordBehavior: "behavior",
        CoordOutcome: "outcome",
    }

    def __init__(self, *factors: ExpFactor) -> None:
        #  Initialize all the factors to None
        self.task = None
        self.attention = None
        self.category = None
        self.behavior = None
        self.outcome = None
        #  Assign the provided factors to the corresponding attributes
        for factor in factors:
            attr_name = self.FACTOR_TO_ATTR.get(factor.__class__)
            if attr_name is None:
                raise ValueError(f"Unsupported factor class: {factor.__class__}")
            setattr(self, attr_name, factor)

    def __repr__(self) -> str:
        not_none_fact = {
            attr: getattr(self, attr)
            for attr in self.FACTOR_TO_ATTR.values()
            if getattr(self, attr)
        }
        return f"ExpCondition({', '.join(f'{name}={val}' for name, val in not_none_fact.items())})"

    @staticmethod
    def combine_factors(*selected_factors: Sequence[ExpFactor] | ExpFactor) -> List["ExpCondition"]:
        """
        Generate experimental conditions by combining a set of experimental factors.

        Arguments
        ---------
        selected_factors : Sequence[ExpFactor] | ExpFactor
            Instances of the experimental factors to consider to generate the experimental
            conditions of interest. Each factor must match a recognized type in
            `ExpCondition.FACTOR_TO_ATTR`.

        Returns
        -------
        exp_conds : List[ExpCondition]
            Experimental conditions generated from the Cartesian product of the selected factor
            instances.

        Examples
        --------
        Generate all possible conditions for two tasks, a fixed attentional state, two stimuli and
        both behaviors:

        >>> exp_conds = ExpCondition.generate_conditions(
        ...     tasks=[Task("PTD"), Task("CLK")],
        ...     attentions=Attention("a"),
        ...     categories=[Category("R"), Category("T")],
        ... )
        >>> exp_conds
        [ExpCondition(task=PTD, attention=a, category=R),
         ExpCondition(task=PTD, attention=a, category=T),
         ExpCondition(task=CLK, attention=a, category=R),
         ExpCondition(task=CLK, attention=a, category=T)]

        Notes
        -----
        For each factor, if a single instance is provided, it considered as fixed and will be used
        for all the conditions.
        If no instance is provided, it is set to `None` in all the experimental conditions.
        To retrieve all the allowed values for experimental factor, use the `get_factors` method of
        the factor class.

        See Also
        --------
        `itertools.product`: Cartesian product of input iterables.
        """
        # Format to list for consistency
        sequences = [[fact] if isinstance(fact, ExpFactor) else fact for fact in selected_factors]
        # Generate all combinations of the provided factors
        # Output: List[Tuple[ExpFactor, ...]]
        combinations = product(*sequences)
        # Create and return a list of ExpCondition instances from the combinations
        return [ExpCondition(*comb) for comb in combinations]

    def __add__(self, other: Self) -> "ExpConditionUnion":
        """
        Combine two conditions into a union of conditions.

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task='PTD', attention='a', category='R')
        >>> exp_cond_2 = ExpCondition(task='CLK', attention='p', category='T')
        >>> union = exp_cond_1 + exp_cond_2
        >>> union.conditions
        [ExpCondition(task='PTD', attention='a', category='R'),
         ExpCondition(task='CLK', attention='p', category='T')]
        """
        return ExpConditionUnion([self, other])

    @classmethod
    def union(cls, *exp_conds: Self) -> "ExpConditionUnion":
        """
        Combine an arbitrary number of conditions into a union.

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task='PTD', attention='a', category='R')
        >>> exp_cond_2 = ExpCondition(task='CLK', attention='p', category='T')
        >>> exp_cond_3 = ExpCondition(task='PTD', category='R', behavior='Go')
        >>> union = ExpCondition.union(exp_cond_1, exp_cond_2, exp_cond_3)
        >>> union.get()
        [ExpCondition(task='PTD', attention='a', category='R'),
         ExpCondition(task='CLK', attention='p', category='T'),
         ExpCondition(task='PTD', category='R', behavior='Go')]
        """
        return ExpConditionUnion(exp_conds)

    def match(self, *coords: Coordinate) -> np.ndarray:
        """
        Generate a boolean mask to index a set of samples based on a condition.

        Arguments
        ---------
        coords : Coordinate
            Coordinates of task, attentional state, category, behavior and/or outcome labels for
            each sample.

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match the condition.
            .. _mask:

        Raises
        ------
        ValueError
            If no coordinate is provided to match the condition.
            If the coordinate class is not valid.

        Examples
        --------
        >>> exp_cond = ExpCondition(task='PTD', attention='a', category='R')
        >>> task_coord = np.array(['PTD', 'CLK', 'PTD'])
        >>> attn_coord = np.array(['a', 'p', 'a'])
        >>> categ_coord = np.array(['R', 'T', 'R'])
        >>> mask = exp_cond.match(task=task_coord, attn=attn_coord, categ=categ_coord)
        >>> mask
        array([ True, False,  True])

        Notes
        -----
        For each factor coordinate, the attribute to use for comparison is the one specified in the
        mapping `ExpCondition.COORD_TO_ATTR`.

        """
        # Initialize the mask with all True values
        if not coords:
            raise ValueError("No coordinate provided to match the condition.")
        mask = np.ones(len(coords[0]), dtype=bool)
        # Apply matching conditions for each coordinate only for specified instance factors
        for coord in coords:
            coord_class = coord.__class__
            attr_name = self.COORD_TO_ATTR.get(coord_class)  # corresponding attribute
            if attr_name is None:
                raise ValueError(f"Unsupported coordinate class: {coord.__class__}")
            match_value = getattr(self, attr_name)
            if match_value is not None:
                mask &= coord == match_value
        return mask

    def count(self, *coords: Coordinate) -> int:
        """
        Count the number of samples that match the condition.

        Arguments
        ---------
        coords : Coordinate

        Returns
        -------
        n_samples : int
            Number of samples matching the condition.
        """
        mask = self.match(*coords)
        return np.sum(mask)


class ExpConditionUnion:
    """
    Union of experimental conditions.

    Attributes
    ----------
    exp_conds : List[ExpCondition]
        Union of conditions represented under the form of a list.

    Methods
    -------
    `get`
    `match`

    Notes
    -----
    Initialization is done by passing an arbitrary number of condition objects.
    In practice, this class is not meant to be instantiated directly, but rather through the `+`
    operator or the `union` class method of `ExpCondition`.
    """

    def __init__(self, exp_conds: Iterable[ExpCondition]) -> None:
        self.exp_conds = list(exp_conds)

    def get(self) -> List[ExpCondition]:
        """Get the list of conditions in the union."""
        return self.exp_conds

    def match(self, **coords: Coordinate) -> np.ndarray:
        """
        Generate a boolean mask to index samples that match any of the conditions.

        Arguments
        ---------
        coords : Coordinate

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match any of the conditions.

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task='PTD', attention='a', category='R')
        >>> exp_cond_2 = ExpCondition(task='CLK', attention='a', category='T')
        >>> union = exp_cond_1 + exp_cond_2
        >>> task_coord = np.array(['PTD', 'CLK', 'PTD'])
        >>> attn_coord = np.array(['a', 'p', 'a'])
        >>> categ_coord = np.array(['R', 'T', 'R'])
        >>> mask = union.match(task=task_coord, attn=attn_coord, categ=categ_coord)
        >>> mask
        array([ True, False,  True])
        """
        # Get the masks for each condition in the union using the match method of ExpCondition
        all_masks = [cond.match(**coords) for cond in self.exp_conds]
        # Combine all masks with a logical OR operation
        mask = np.logical_or.reduce(all_masks)
        return mask

    def count(self, **coords: Coordinate) -> int:
        """
        Count the number of samples that match any of the conditions.

        Arguments
        ---------
        coords : Coordinate

        Returns
        -------
        n_samples : int
            Number of samples matching any of the conditions.
        """
        mask = self.match(**coords)
        return np.sum(mask)
