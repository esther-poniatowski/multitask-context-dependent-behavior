#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.exp_features` [module]

Classes representing the experimental features describing the behavioral paradigm.

Classes
-------
`ExpFactor`
`Task`
`Attention`
`Category`
`Stimulus`
`Behavior`
"""

from types import MappingProxyType
from typing import FrozenSet, Self, List

from core.entities.base_entity import Entity


class ExpFactor(str, Entity[str]):
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

    @classmethod
    def get_features(cls) -> List[Self]:
        """Return the list of all features instances corresponding to the options."""
        return [cls(option) for option in cls.OPTIONS]


class Task(ExpFactor):
    """
    Tasks performed by the animals, defined by the two sounds categories to discriminate.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Tasks represented by their short names (3 letters):

        - 'PTD': Pure Tone Discrimination. Stimuli: Pure tone vs. White noise.
        - 'CLK': Click Rate Discrimination. Stimuli: Click trains at fast and slow rates.
        - 'CCH': Complex Chord Discrimination. Stimuli: Two complex chords. Similar to 'PTD' in
          terms of task structure.
    """

    OPTIONS = frozenset(["PTD", "CLK", "CCH"])
    LABELS = MappingProxyType(
        {
            "PTD": "Pure Tone Discrimination",
            "CLK": "Click Rate Discrimination",
            "CCH": "Complex Chord Discrimination",
        }
    )


class Attention(ExpFactor):
    """
    Attentional state, defined bh the animals' engagement in a task (by assumption).

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Attentional states represented by their short names (1 to 4 letters):

        - 'a': Active context (only for trained animals). The animal engages in an aversive Go/No-Go
          task. It is presented with a licking spout and should refrain from licking after the
          presentation of a target stimulus.
        - 'p': Passive context (for both trained and naive animals). The animal only listens to
          sounds without the possibility to lick.
        - 'p-pre': Pre-passive context (only for trained animals). Sup-type of 'p': Passive session
          before an active session.
        - 'p-post': Post-passive context (only for trained animals). Sup-type of 'p': Passive
          session after an active session.

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


class Category(ExpFactor):
    """
    Behavioral category of the stimuli in the Go/NoGo task.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Stimuli represented by their short names (1 letter):

        - 'R': Reference stimulus, i.e. Go (safe).
          In PTD : TORC sound.
          In CLK : Click train at either fast or slow rate.
        - 'T': Target stimulus, i.e. NoGo (dangerous).
          In PTD : Pure tone.
          In CLK : Click train at the other rate.
        - 'N': Neutral stimulus.
          In CLK (only) : TORC sound.

    Methods
    -------
    `get_clk`
    `get_ptd`

    Notes
    -----
    The same stimuli labels are used for consistency in both active and passive contexts, even
    though they are meaningful only in the active attentional state. Moreover, the exact nature of the sound
    under a given label differ across tasks.

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


class Stimulus(ExpFactor):
    """
    Sensory (physical) nature of the auditory stimuli presented to the animals.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Stimuli represented by their short names (3 to 4 letters):

        - 'TORC': Time-Orthogonal Ripple Counter, noise-like sound. Either a reference in PTD or a
          neutral stimulus in CLK.
        - 'Tone': Pure tone. Always a target in PTD.
        - 'ClickT': Click train at a given rate. Always a target in CLK.
        - 'ClickR': Click train at a different rate. Always a reference in CLK.

    Notes
    -----
    To refer to both categories of click rates, the labels "ClickT" and "ClickR" are used since
    the exact rates associated to targets and references differ across animals in the CLK task.
    """

    OPTIONS = frozenset(["TORC", "Click", "Tone", "Noise"])
    LABELS = MappingProxyType(
        {
            "TORC": "Time-Orthogonal Ripple Counter",
            "Click": "Click Train",
            "Tone": "Pure Tone",
            "Noise": "White Noise",
        }
    )


class Behavior(ExpFactor):
    """
    Behavioral choice, i.e. animals' response.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Behavioral choices represented by their short names (2 to 4 letters):

        - 'Go': Lick.
        - 'NoGo': Refrain from licking.
    """

    OPTIONS = frozenset(["Go", "NoGo"])
    LABELS = MappingProxyType({"Go": "Lick", "NoGo": "No Lick"})


class EventDescription(ExpFactor):
    """
    Valid descriptions of events in the raw data.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Event descriptions (4 to 9 letters):

        - 'TRIALSTART': Marks the start of a block.
        - 'PreStimSilence': Marks the pre-stimulus silence period within a slot.
        - 'Stim': Marks the presentation of a stimulus within a slot.
        - 'PostStimSilence': Marks the post-stimulus silence period within a slot.
        - 'BEHAVIOR,SHOCKON': Marks the occurrence of a shock on an error trial.
        - 'TRIALSTOP': Marks the end of a block.
    """

    OPTIONS = frozenset(
        ["TRIALSTART", "PreStimSilence", "Stim", "PostStimSilence", "BEHAVIOR,SHOCKON", "TRIALSTOP"]
    )
    LABELS = MappingProxyType(
        {
            "TRIALSTART": "Block Start",
            "PreStimSilence": "Pre-Stimulus Silence",
            "Stim": "Stimulus Presentation",
            "PostStimSilence": "Post-Stimulus Silence",
            "BEHAVIOR,SHOCKON": "Shock Delivery",
            "TRIALSTOP": "Block Stop",
        }
    )
