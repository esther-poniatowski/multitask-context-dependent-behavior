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

from core.entities.exp_features import Task, Context, Stimulus, Behavior, ExpFeature

if TYPE_CHECKING:  # prevent circular imports (`exp_condition` is imported coordinate modules)
    from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim, CoordOutcome


class ExpCondition:
    """
    Experimental condition defined by the combination of several experimental features, among:

    - task
    - context
    - stimulus
    - behavior outcome

    Attributes
    ----------
    task : Task
        Task performed by the animal.
    context : Context
        Context in which the animal is engaged.
    stimulus : Stimulus
        Stimulus category presented to the animal.
    outcome : Behavior
        Outcome of the animal's behavior.

    Methods
    -------
    `match`
    `get_combinations`

    Notes
    -----
    Partial conditions can be defined by specifying only the features of interest. The remaining
    features are set to `None` by default.

    Warning
    -------
    Instantiate with keyword arguments to avoid confusion between the features, especially when
    partial conditions are defined.

    Examples
    --------
    Initialize a condition for the task PTD, in the active context, with the reference stimulus:

    >>> task = Task('PTD')
    >>> context = Context('a')
    >>> stimulus = Stimulus('R')
    >>> condition1 = ExpCondition(task=task, context=context, stimulus=stimulus)
    >>> condition1.task
    'PTD'

    Initialize directly with strings:

    >>> condition2 = ExpCondition(task='CLK', context='p', stimulus='T')

    Initialize a partial condition without specifying the context:

    >>> condition3 = ExpCondition(task='PTD', stimulus='R', outcome='Go')
    >>> print(condition3)
    ExpCondition(task=PTD, context=None, stimulus=R, outcome=Go)

    Combine two conditions into a union:

    >>> union = condition1 + condition2
    >>> union.conditions
    [ExpCondition(task='PTD', context='a', stimulus='R'), ExpCondition(task='CLK', context='p', stimulus='T')]

    Combine an arbitrary number of conditions into a union:

    >>> union = ExpCondition.union(condition1, condition2, condition3)
    >>> union.conditions
    [ExpCondition(task='PTD', context='a', stimulus='R'), ExpCondition(task='CLK', context='p', stimulus='T'), ExpCondition(task='PTD', context=None, stimulus='R', outcome='Go')]

    Filter samples based on a condition:

    >>> task_coord = np.array(['PTD', 'CLK', 'PTD'])
    >>> ctx_coord = np.array(['a', 'p', 'a'])
    >>> stim_coord = np.array(['R', 'T', 'R'])
    >>> mask = condition1.match(task=task_coord, ctx=ctx_coord, stim=stim_coord)
    >>> mask
    array([ True, False,  True])

    See Also
    --------
    `Task`
    `Context`
    `Stimulus`
    `Behavior`
    `ExpConditionUnion`
    """

    def __init__(
        self,
        task: Task | str | None = None,
        context: Context | str | None = None,
        stimulus: Stimulus | str | None = None,
        outcome: Behavior | str | None = None,
    ):
        if isinstance(task, str):
            task = Task(task)
        if isinstance(context, str):
            context = Context(context)
        if isinstance(stimulus, str):
            stimulus = Stimulus(stimulus)
        if isinstance(outcome, str):
            outcome = Behavior(outcome)
        self.task = task
        self.context = context
        self.stimulus = stimulus
        self.outcome = outcome

    def __repr__(self) -> str:
        return (
            f"ExpCondition("
            f"task={self.task}, "
            f"context={self.context}, "
            f"stimulus={self.stimulus}, "
            f"outcome={self.outcome})"
        )

    def match(
        self,
        task: "CoordTask" | None = None,
        ctx: "CoordCtx" | None = None,
        stim: "CoordStim" | None = None,
        outcome: "CoordOutcome" | None = None,
    ) -> np.ndarray:
        """
        Generate a boolean mask to index a set of samples based on a condition.

        Arguments
        ---------
        task, ctx, stim, outcome : Coordinate
            Coordinates of task, context, stimulus and behavior outcome labels for each sample.
            .. _task_coord:
            .. _ctx_coord:
            .. _stim_coord:
            .. _outcome_coord:

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match the condition.
            .. _mask:
        """
        # Identify the specified coordinates to filter (not None)
        coords = [c for c in (task, ctx, stim, outcome) if c is not None]
        if not coords:
            raise ValueError("No coordinate provided to match the condition.")
        # Retrieve the number of samples from one specified coordinate
        mask = np.ones(len(coords[0]), dtype=bool)
        # Apply conditions only for specified instance feature and corresponding coordinate
        if self.task and task is not None:
            mask &= task == self.task
        if self.context and ctx is not None:
            mask &= ctx == self.context
        if self.stimulus and stim is not None:
            mask &= stim == self.stimulus
        if self.outcome and outcome is not None:
            mask &= outcome == self.outcome
        return mask

    def __add__(self, other: Self) -> "ExpConditionUnion":
        """
        Combine two conditions into a union of conditions.
        """
        return ExpConditionUnion([self, other])

    @classmethod
    def union(cls, *conditions: Self) -> "ExpConditionUnion":
        """
        Combine an arbitrary number of conditions into a union.
        """
        return ExpConditionUnion(conditions)

    @staticmethod
    def get_combinations(*features) -> List:
        """
        Get all possible combinations of the values of experimental features of interest.

        Arguments
        ---------
        features : Tuple[ExpFeature]
            Classes of the experimental features to consider, among: `Task`, `Context`, `Stimulus`,
            `Behavior`.

        Returns
        -------
        combinations : List[Tuple[ExpFeature, ...]]
            Cartesian product of feature instances in the selected experimental features.

        Notes
        -----
        The values of each experimental feature are retrieved by the `get_features` method of the
        features classes.

        Examples
        --------
        Get all possible combinations of task, context, and stimulus:

        >>> combinations = ExpCondition.get_combinations(Task, Stimulus, Behavior)

        Get all possible combinations of task, stimulus and behavior outcome:

        >>> combinations = ExpCondition.get_combinations(Task, Stimulus, Behavior)

        See Also
        --------
        `itertools.product`: Cartesian product of input iterables.
        """
        feature_values = [feature_class.get_features() for feature_class in features]
        combinations = list(product(*feature_values))
        return combinations

    @staticmethod
    def generate_conditions(
        tasks: Sequence[Task] | Task | None = None,
        stimuli: Sequence[Stimulus] | Stimulus | None = None,
        contexts: Sequence[Context] | Context | None = None,
        outcomes: Sequence[Behavior] | Behavior | None = None,
    ) -> List["ExpCondition"]:
        """
        Generate all possible conditions for a set of experimental features.

        Arguments
        ---------
        tasks, stimuli, contexts, outcomes : List[ExpFeature] | ExpFeature, optional
            Instances of tasks, contexts, stimuli, and behavior outcomes to consider to generate the
            experimental conditions of interest.

        Returns
        -------
        conditions : List[ExpCondition]
            Conditions generated from the Cartesian product of the selected features.

        Examples
        --------
        Generate all possible conditions for two tasks, a fixed context, two stimuli and both
        outcomes:

        >>> conditions = ExpCondition.generate_conditions(
        ...     tasks=[Task("PTD"), Task("CLK")],
        ...     contexts=[Context("a")],
        ...     stimuli=[Stimulus("R"), Stimulus("T")],
        ...     outcomes=[Behavior("Go"), Behavior("NoGo")]
        ... )

        Notes
        -----
        For each feature, if a single instance is provided, it considered as fixed and will be used
        for all conditions.
        If no instance is provided, it is set to `None` in all the experimental conditions.
        """
        # Ensure each argument is a list or None
        tasks_seq = ExpCondition.format_to_combine(tasks)
        stimuli_seq = ExpCondition.format_to_combine(stimuli)
        contexts_seq = ExpCondition.format_to_combine(contexts)
        outcomes_seq = ExpCondition.format_to_combine(outcomes)
        # Generate all combinations of the provided features
        # Output: List[Tuple[ExpFeature, ...]]
        combinations = product(tasks_seq, contexts_seq, stimuli_seq, outcomes_seq)
        # Create and return a list of ExpCondition instances from the combinations
        return [ExpCondition(t, c, s, o) for t, c, s, o in combinations]

    @staticmethod
    def format_to_combine(
        arg: Sequence[ExpFeature] | ExpFeature | None,
    ) -> Sequence[ExpFeature | None]:
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
        if isinstance(arg, list) and all(isinstance(a, ExpFeature) or a is None for a in arg):
            return arg
        elif isinstance(arg, ExpFeature) or arg is None:
            return [arg]
        else:
            raise ValueError(f"Invalid argument type: {type(arg)}")


class ExpConditionUnion:
    """
    Union of experimental conditions.

    Attributes
    ----------
    conditions : List[ExpCondition]
        Union of conditions represented under the form of a list.

    Notes
    -----
    Initialization is done by passing an arbitrary number of condition objects.
    In practice, this class is not meant to be instantiated directly, but rather through the `+`
    operator or the `union` class method of `ExpCondition`.
    """

    def __init__(self, conditions: Iterable[ExpCondition]) -> None:
        self.conditions = list(conditions)

    def match(
        self,
        task: "CoordTask" | None = None,
        ctx: "CoordCtx" | None = None,
        stim: "CoordStim" | None = None,
        outcome: "CoordOutcome" | None = None,
    ) -> np.ndarray:
        """
        Generate a boolean mask to index samples that match any of the conditions.

        Arguments
        ---------
        task, ctx, stim, outcome : np.ndarray
            See :ref:`task`, :ref:`ctx`, and :ref:`stim`.

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match any of the conditions.
        """
        # Get the masks for each condition in the union using the match method of ExpCondition
        all_masks = [condition.match(task, ctx, stim, outcome) for condition in self.conditions]
        # Combine all masks with a logical OR operation
        mask = np.logical_or.reduce(all_masks)
        return mask
