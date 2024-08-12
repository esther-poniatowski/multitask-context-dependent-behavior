#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.core_objects.exp_condition` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
:class:`Task`
:class:`Context`
:class:`Stimulus`
"""

from types import MappingProxyType
from typing import FrozenSet, Mapping

from core.core_objects.base import CoreObject


class Task(CoreObject[str]):
    """
    Tasks performed by the animals (i.e. discrimination between two sounds categories).

    Class Attributes
    ----------------
    _options : FrozenSet[str]
        Tasks represented by their short names (3 letters).
        PTD : Pure Tone Discrimination.
            Categories : Pure tone vs. White noise.
        CLK : Click Rate Discrimination.
            Categories : Click trains at fast and slow rates.
        CCH : sComplex Chord Discrimination.
            Categories : Two complex chords.
            Similar to PTD in terms of task structure.
    """
    _options: FrozenSet[str] = frozenset(["PTD", "CLK", "CCH"])

    _full_labels: Mapping[str, str] = MappingProxyType({
        "PTD": "Pure Tone Discrimination",
        "CLK": "Click Rate Discrimination",
        "CCH": "Complex Chord Discrimination"
    })


class Context(CoreObject[str]):
    """
    Contexts, i.e. animals' engagement in a task.

    Class Attributes
    ----------------
    _options : FrozenSet[str]
        Contexts represented by their short names (1 to 4 letters).
        a : Active context (only for trained animals).
            The animal performs an aversive Go/No-Go task.
            It is presented with a licking spout and should refrain from licking
            after the presentation of a target stimulus.
        p : Passive context (for both trained and naive animals).
            The animal only listens to sounds without licking.
        p-pre : Pre-passive context (only for trained animals).
            It corresponds to a passive session before an active session.
        p-post : Post-passive context (only for trained animals).
            It corresponds to a passive session after an active session.

    Methods
    -------
    :meth:`get_trained`
    :meth:`get_naive`
    """
    _options: FrozenSet[str] = frozenset(["a", "p", "p-pre", "p-post"])

    @classmethod
    def get_trained(cls) -> FrozenSet['Context']:
        """Contexts for trained animals (all)."""
        return frozenset(cls(o) for o in cls._options)

    @classmethod
    def get_naive(cls) -> FrozenSet['Context']:
        """Contexts for naive animals (only passive)."""
        return frozenset([cls("p")])

    _full_labels: Mapping[str, str] = MappingProxyType({
        "p": "Passive",
        "a": "Active",
        "p-pre": "Pre-Passive",
        "p-post": "Post-Passive"
    })


class Stimulus(CoreObject[str]):
    """
    Behavioral category of the auditory stimuli in the Go/NoGo task.

    Notes
    -----
    The same stimuli labels are used for consistency in both active and passive contexts,
    even though they are meaningful only in the active context.
    Moreover, the exact nature of the sound under a given label differ across tasks.

    Class Attributes
    ----------------
    _options : FrozenSet[str]
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
    :meth:`get_clk`
    :meth:`get_ptd`
    """
    _options: FrozenSet[str] = frozenset(["R", "T", "N"])

    @classmethod
    def get_clk(cls) -> FrozenSet['Stimulus']:
        """Stimuli for task CLK (all)."""
        return frozenset(cls(o) for o in cls._options)

    @classmethod
    def get_ptd(cls) -> FrozenSet['Stimulus']:
        """Stimuli for task PTD (no neutral)."""
        return frozenset([cls("R"), cls("T")])

    _full_labels: Mapping[str, str] = MappingProxyType({
        "R": "Reference",
        "T": "Target",
        "N": "Neutral"
    })
