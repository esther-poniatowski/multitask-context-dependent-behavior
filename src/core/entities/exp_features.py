#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.exp_condition` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
`ExpFeature`
`Task`
`Context`
`Stimulus`
`ExpCondition`
"""

from types import MappingProxyType
from typing import FrozenSet, Self, Optional, Iterable, Union, TYPE_CHECKING

import numpy as np

from core.entities.base_entity import Entity

if TYPE_CHECKING:  # prevent circular imports (`exp_condition` is imported coordinate modules)
    from core.coordinates.exp_condition import CoordTask, CoordCtx, CoordStim


class ExpFeature(str, Entity[str]):
    """
    Base class for experimental conditions in the behavioral paradigm, behaving like strings.

    Define the `__new__` method to inherit from `str`.

    Subclasses should define their own class-level attributes `OPTIONS` and `LABELS` (if applicable)
    and their own methods (in addition to the base `Entity` methods).

    See Also
    --------
    `Entity.is_valid`
    """

    def __new__(cls, value: str) -> Self:
        if not cls.is_valid(value):  # method from Entity
            raise ValueError(f"Invalid value for {cls.__name__}: {value}.")
        return super().__new__(cls, value)


class Task(ExpFeature):
    """
    Tasks performed by the animals (i.e. discrimination between two sounds categories).

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Tasks represented by their short names (3 letters).
        PTD : Pure Tone Discrimination.
            Categories : Pure tone vs. White noise.
        CLK : Click Rate Discrimination.
            Categories : Click trains at fast and slow rates.
        CCH : sComplex Chord Discrimination.
            Categories : Two complex chords.
            Similar to PTD in terms of task structure.
    """

    OPTIONS = frozenset(["PTD", "CLK", "CCH"])

    LABELS = MappingProxyType(
        {
            "PTD": "Pure Tone Discrimination",
            "CLK": "Click Rate Discrimination",
            "CCH": "Complex Chord Discrimination",
        }
    )


class Context(ExpFeature):
    """
    Contexts, i.e. animals' engagement in a task.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Contexts represented by their short names (1 to 4 letters).
        a : Active context (only for trained animals).
            The animal performs an aversive Go/No-Go task.
            It is presented with a licking spout and should refrain from licking after the
            presentation of a target stimulus.
        p : Passive context (for both trained and naive animals).
            The animal only listens to sounds without licking.
        p-pre : Pre-passive context (only for trained animals).
            It corresponds to a passive session before an active session.
        p-post : Post-passive context (only for trained animals).
            It corresponds to a passive session after an active session.

    Methods
    -------
    `get_trained`
    `get_naive`
    """

    OPTIONS = frozenset(["a", "p", "p-pre", "p-post"])

    LABELS = MappingProxyType(
        {"p": "Passive", "a": "Active", "p-pre": "Pre-Passive", "p-post": "Post-Passive"}
    )

    @classmethod
    def get_trained(cls) -> FrozenSet[Self]:
        """Contexts for trained animals (all)."""
        return frozenset(cls(o) for o in cls.OPTIONS)

    @classmethod
    def get_naive(cls) -> FrozenSet[Self]:
        """Contexts for naive animals (only passive)."""
        return frozenset([cls("p")])


class Stimulus(ExpFeature):
    """
    Behavioral category of the auditory stimuli in the Go/NoGo task.

    Notes
    -----
    The same stimuli labels are used for consistency in both active and passive contexts, even
    though they are meaningful only in the active context. Moreover, the exact nature of the sound
    under a given label differ across tasks.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Stimuli represented by their short names (1 letter).
        R : Reference stimulus, i.e. Go (safe).
            In PTD : TORC sound.
            In CLK : Click train at either fast or slow rate.
        T : Target stimulus, i.e. NoGo (dangerous).
            In PTD : Pure tone.
            In CLK : Click train at the other rate.
        N : Neutral stimulus.
            In CLK (only) : TORC sound.

    Methods
    -------
    `get_clk`
    `get_ptd`
    """

    OPTIONS = frozenset(["R", "T", "N"])
    LABELS = MappingProxyType({"R": "Reference", "T": "Target", "N": "Neutral"})

    @classmethod
    def get_clk(cls) -> FrozenSet[Self]:
        """Stimuli for task CLK (all)."""
        return frozenset(cls(o) for o in cls.OPTIONS)

    @classmethod
    def get_ptd(cls) -> FrozenSet[Self]:
        """Stimuli for task PTD (no neutral)."""
        return frozenset([cls("R"), cls("T")])


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
