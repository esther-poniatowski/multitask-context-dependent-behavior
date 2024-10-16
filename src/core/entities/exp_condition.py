#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.exp_condition` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
`Task`
`Context`
`Stimulus`
"""

from types import MappingProxyType
from typing import FrozenSet, Self

from entities.base_entity import Entity


class ExpFeature(str, Entity[str]):
    """
    Experimental conditions in the behavioral paradigm, behaving like strings.

    Define the `__new__` method to inherit from `str`.

    Subclasses should define their own class-level attributes `OPTIONS` and `LABELS` (if applicable)
    and their own methods (in addition to the base `Entity` methods).

    See Also
    --------
    :meth:`Entity.is_valid`
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
