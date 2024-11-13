#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.exp_conditions` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
`ExpCondition`
"""

from typing import Self, Iterable, List, Sequence, TYPE_CHECKING
from itertools import product

import numpy as np

from core.entities.exp_factors import Task, Attention, Category, Behavior, ExpFactor

if TYPE_CHECKING:  # prevent circular imports (`exp_condition` is imported coordinate modules)
    from coordinates.exp_factor_coord import (
        CoordTask,
        CoordAttention,
        CoordStimulus,
        CoordCategory,
        CoordBehavior,
    )


class ExpCondition:
    """
    Experimental condition defined by the combination of several experimental factors, among:

    - task
    - attentional state
    - stimulus category
    - behavioral choice

    Attributes
    ----------
    task : Task
        Task performed by the animal.
    attention : Attention
        Attentional state in which the animal is engaged.
    category : Category
        Category of the stimulus presented to the animal.
    behavior : Behavior
        Behavioral choice of the animal.

    Methods
    -------
    `match`
    `get_combinations`

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
    Initialize a condition for the task PTD, in the active attention, with the reference stimulus:

    >>> task = Task('PTD')
    >>> attention = Attention('a')
    >>> category = Category('R')
    >>> exp_cond_1 = ExpCondition(task=task, attention=attention, category=category)
    >>> exp_cond_1.task
    'PTD'

    Initialize directly with strings:

    >>> exp_cond_2 = ExpCondition(task='CLK', attention='p', category='T')

    Initialize a partial condition without specifying the attention:

    >>> exp_cond_3 = ExpCondition(task='PTD', category='R', behavior='Go')
    >>> exp_cond_3
    ExpCondition(task=PTD, attention=None, category=R, behavior=Go)

    Combine two conditions into a union:

    >>> union = exp_cond_1 + exp_cond_2
    >>> union.conditions
    [ExpCondition(task='PTD', attention='a', category='R'), ExpCondition(task='CLK', attention='p', category='T')]

    Combine an arbitrary number of conditions into a union:

    >>> union = ExpCondition.union(exp_cond_1, exp_cond_2, exp_cond_3)
    >>> union.conditions
    [ExpCondition(task='PTD', attention='a', category='R'), ExpCondition(task='CLK', attention='p', category='T'), ExpCondition(task='PTD', attention=None, category='R', behavior='Go')]

    Filter samples based on a condition:

    >>> task_coord = np.array(['PTD', 'CLK', 'PTD'])
    >>> attn_coord = np.array(['a', 'p', 'a'])
    >>> categ_coord = np.array(['R', 'T', 'R'])
    >>> mask = condition1.match(task=task_coord, attn=attn_coord, categ=categ_coord)
    >>> mask
    array([ True, False,  True])

    See Also
    --------
    `Task`
    `Attention`
    `Stimulus`
    `Behavior`
    `ExpConditionUnion`
    """

    def __init__(
        self,
        task: Task | str | None = None,
        attention: Attention | str | None = None,
        category: Category | str | None = None,
        behavior: Behavior | str | None = None,
    ):
        if isinstance(task, str):
            task = Task(task)
        if isinstance(attention, str):
            attention = Attention(attention)
        if isinstance(category, str):
            category = Category(category)
        if isinstance(behavior, str):
            behavior = Behavior(behavior)
        self.task = task
        self.attention = attention
        self.category = category
        self.behavior = behavior

    def __repr__(self) -> str:
        return (
            f"ExpCondition("
            f"task={self.task}, "
            f"attention={self.attention}, "
            f"category={self.category}, "
            f"behavior={self.behavior})"
        )

    def match(
        self,
        task: "CoordTask" | None = None,
        attn: "CoordAttention" | None = None,
        categ: "CoordCategory" | None = None,
        behav: "CoordBehavior" | None = None,
    ) -> np.ndarray:
        """
        Generate a boolean mask to index a set of samples based on a condition.

        Arguments
        ---------
        task, attn, categ, behavior : Coordinate
            Coordinates of task, attentional state, stimulus and behavior labels for each sample.
            .._task_coord:
            .. _attn_coord:
            .. _stim_coord:
            .. _behav_coord:

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match the condition.
            .. _mask:
        """
        # Identify the specified coordinates to filter (not None)
        coords = [c for c in (task, attn, categ, behav) if c is not None]
        if not coords:
            raise ValueError("No coordinate provided to match the condition.")
        # Retrieve the number of samples from one specified coordinate
        mask = np.ones(len(coords[0]), dtype=bool)
        # Apply conditions only for specified instance factor and corresponding coordinate
        if self.task and task is not None:
            mask &= task == self.task
        if self.attention and attn is not None:
            mask &= attn == self.attention
        if self.category and categ is not None:
            mask &= categ == self.category
        if self.behavior and behav is not None:
            mask &= behav == self.behavior
        return mask

    def __add__(self, other: Self) -> "ExpConditionUnion":
        """
        Combine two conditions into a union of conditions.
        """
        return ExpConditionUnion([self, other])

    @classmethod
    def union(cls, *exp_conds: Self) -> "ExpConditionUnion":
        """
        Combine an arbitrary number of conditions into a union.
        """
        return ExpConditionUnion(exp_conds)

    @staticmethod
    def get_combinations(*factors) -> List:
        """
        Get all possible combinations of the values of experimental factors of interest.

        Arguments
        ---------
        factors : Tuple[ExpFactor]
            Classes of the experimental factors to consider, among: `Task`, `Attention`, `Category`,
            `Behavior`.

        Returns
        -------
        combinations : List[Tuple[ExpFactor, ...]]
            Cartesian product of factor instances in the selected experimental factors.

        Notes
        -----
        The values of each experimental factor are retrieved by the `get_factors` method of the
        factors classes.

        Examples
        --------
        Get all possible combinations of task, attentional state, and category:

        >>> combinations = ExpCondition.get_combinations(Task, Attention, Category)

        See Also
        --------
        `itertools.product`: Cartesian product of input iterables.
        """
        factor_values = [factor_class.get_factors() for factor_class in factors]
        combinations = list(product(*factor_values))
        return combinations

    @staticmethod
    def generate_conditions(
        tasks: Sequence[Task] | Task | None = None,
        attentions: Sequence[Attention] | Attention | None = None,
        categories: Sequence[Category] | Category | None = None,
        behaviors: Sequence[Behavior] | Behavior | None = None,
    ) -> List["ExpCondition"]:
        """
        Generate all possible conditions for a set of experimental factors.

        Arguments
        ---------
        tasks, attentions, categories behaviors : List[ExpFactor] | ExpFactor, optional
            Instances of tasks, attentions, categories and behaviors to consider to generate the
            experimental conditions of interest.

        Returns
        -------
        exp_conds : List[ExpCondition]
            Conditions generated from the Cartesian product of the selected factors.

        Examples
        --------
        Generate all possible conditions for two tasks, a fixed attentional state, two stimuli and
        both behaviors:

        >>> exp_conds = ExpCondition.generate_conditions(
        ...     tasks=[Task("PTD"), Task("CLK")],
        ...     attentions=[Attention("a")],
        ...     categories=[Category("R"), Category("T")],
        ...     behaviors=[Behavior("Go"), Behavior("NoGo")]
        ... )

        Notes
        -----
        For each factor, if a single instance is provided, it considered as fixed and will be used
        for all conditions.
        If no instance is provided, it is set to `None` in all the experimental conditions.
        """
        # Ensure each argument is a list or None
        tasks_seq = ExpCondition.format_to_combine(tasks)
        categ_seq = ExpCondition.format_to_combine(categories)
        attn_seq = ExpCondition.format_to_combine(attentions)
        behav_seq = ExpCondition.format_to_combine(behaviors)
        # Generate all combinations of the provided factors
        # Output: List[Tuple[ExpFactor, ...]]
        combinations = product(tasks_seq, attn_seq, categ_seq, behav_seq)
        # Create and return a list of ExpCondition instances from the combinations
        return [ExpCondition(t, a, c, b) for t, a, c, b in combinations]

    @staticmethod
    def format_to_combine(
        arg: Sequence[ExpFactor] | ExpFactor | None,
    ) -> Sequence[ExpFactor | None]:
        """
        Helper function to ensure each argument is a list or None.

        Arguments
        ---------
        arg : Any
            Argument to ensure is a list or None.

        Returns
        -------
        arg : List[Any]
            If the argument is a list, it is returned as is.
            If it is a single element of None, it is returned as a list with a single element.
        """
        if isinstance(arg, list) and all(isinstance(a, ExpFactor) or a is None for a in arg):
            return arg
        elif isinstance(arg, ExpFactor) or arg is None:
            return [arg]
        else:
            raise ValueError(f"Invalid argument type: {type(arg)}")


class ExpConditionUnion:
    """
    Union of experimental conditions.

    Attributes
    ----------
    exp_conds : List[ExpCondition]
        Union of conditions represented under the form of a list.

    Notes
    -----
    Initialization is done by passing an arbitrary number of condition objects.
    In practice, this class is not meant to be instantiated directly, but rather through the `+`
    operator or the `union` class method of `ExpCondition`.
    """

    def __init__(self, exp_conds: Iterable[ExpCondition]) -> None:
        self.exp_conds = list(exp_conds)

    def match(
        self,
        task: "CoordTask" | None = None,
        attn: "CoordAttention" | None = None,
        categ: "CoordCategory" | None = None,
        behav: "CoordBehavior" | None = None,
    ) -> np.ndarray:
        """
        Generate a boolean mask to index samples that match any of the conditions.

        Arguments
        ---------
        task, attn, categ, behav : np.ndarray
            See :ref:`task`, :ref:`attn`, :ref:`categ`, :ref:`behav`.

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match any of the conditions.
        """
        # Get the masks for each condition in the union using the match method of ExpCondition
        all_masks = [cond.match(task, attn, categ, behav) for cond in self.exp_conds]
        # Combine all masks with a logical OR operation
        mask = np.logical_or.reduce(all_masks)
        return mask
