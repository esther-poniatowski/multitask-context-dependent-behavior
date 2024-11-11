#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.exp_conditions` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
`ExpCondition`
"""

from typing import Self, Optional, Iterable, Union, TYPE_CHECKING

import numpy as np

from core.entities.exp_features import Task, Context, Stimulus

if TYPE_CHECKING:  # prevent circular imports (`exp_condition` is imported coordinate modules)
    from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim


class ExpCondition:
    """
    Experimental condition defined by the combination of a task, a context, and a stimulus.

    Class Attributes
    ----------------
    COMBINATIONS : FrozenSet[Tuple[Task, Context, Stimulus]]
        All possible combinations of task, context, and stimulus (i.e. Cartesian product of the
        three sets of options).

    Attributes
    ----------
    task : Task
        Task performed by the animal.
    context : Context
        Context in which the animal is engaged.
    stimulus : Stimulus
        Stimulus category presented to the animal.

    Notes
    -----
    Partial conditions can be defined by setting some attributes to `None`.

    Examples
    --------
    Initialize a condition for the task PTD, in the active context, with the reference stimulus:

    >>> task = Task('PTD')
    >>> context = Context('a')
    >>> stimulus = Stimulus('R')
    >>> condition = ExpCondition(task, context, stimulus)
    >>> condition.task
    'PTD'

    Initialize directly with strings:

    >>> condition = ExpCondition('PTD', 'a', 'R')

    Initialize a partial condition without specifying the stimulus:

    >>> condition = ExpCondition(task='PTD', context='a')

    Combine two conditions into a union:

    >>> condition1 = ExpCondition('PTD', 'a', 'R')
    >>> condition2 = ExpCondition('CLK', 'p', 'T')
    >>> union = condition1 + condition2
    >>> union.conditions
    [ExpCondition('PTD', 'a', 'R'), ExpCondition('CLK', 'p', 'T')]

    Combine an arbitrary number of conditions into a union:

    >>> union = ExpCondition.union(condition1, condition2)
    >>> union.conditions
    [ExpCondition('PTD', 'a', 'R'), ExpCondition('CLK', 'p', 'T')]

    Filter samples based on a condition:

    >>> task_coord = np.array(['PTD', 'CLK', 'PTD'])
    >>> ctx_coord = np.array(['a', 'p', 'a'])
    >>> stim_coord = np.array(['R', 'T', 'R'])
    >>> mask = condition.match(task_coord, ctx_coord, stim_coord)
    >>> mask
    array([ True, False,  True])

    See Also
    --------
    `Task`
    `Context`
    :`Stimulus`
    `ExpConditionUnion`
    """

    COMBINATIONS = frozenset(
        (task, context, stimulus)
        for task in Task.OPTIONS
        for context in Context.OPTIONS
        for stimulus in Stimulus.OPTIONS
    )

    def __init__(
        self,
        task: Optional[Union[Task, str]] = None,
        context: Optional[Union[Context, str]] = None,
        stimulus: Optional[Union[Stimulus, str]] = None,
    ):
        if isinstance(task, str):
            task = Task(task)
        if isinstance(context, str):
            context = Context(context)
        if isinstance(stimulus, str):
            stimulus = Stimulus(stimulus)
        self.task = task
        self.context = context
        self.stimulus = stimulus

    def match(
        self, task_coord: "CoordTask", ctx_coord: "CoordCtx", stim_coord: "CoordStim"
    ) -> np.ndarray:
        """
        Generate a boolean mask to index a set of samples based on a condition.

        Arguments
        ---------
        task_coord, ctx_coord, stim_coord : np.ndarray
            Arrays of task, context, and stimulus labels for each sample, respectively.
            .. _task_coord:
            .. _ctx_coord:
            .. _stim_coord:

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match the condition.
            .. _mask:
        """
        mask = np.ones(len(task_coord), dtype=bool)
        if self.task:
            mask &= task_coord == self.task
        if self.context:
            mask &= ctx_coord == self.context
        if self.stimulus:
            mask &= stim_coord == self.stimulus
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

    def __repr__(self) -> str:
        return f"ExpCondition(task={self.task}, context={self.context}, stimulus={self.stimulus})"


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

    def match(self, task_coord, ctx_coord, stim_coord) -> np.ndarray:
        """
        Generate a boolean mask to index samples that match any of the conditions.

        Arguments
        ---------
        task_coord, ctx_coord, stim_coord : np.ndarray
            See :ref:`task_coord`, :ref:`ctx_coord`, and :ref:`stim_coord`.

        Returns
        -------
        mask : np.ndarray
            See :ref:`mask`.
        """
        mask = np.zeros(len(task_coord), dtype=bool)
        for condition in self.conditions:
            mask |= condition.match(task_coord, ctx_coord, stim_coord)
        return mask
