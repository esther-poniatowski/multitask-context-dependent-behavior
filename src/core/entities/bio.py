#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.entities.bio` [module]

Classes representing the biological system under investigation.

Classes
-------
:class:`Animal`
:class:`Training`
:class:`Area`
:class:`CorticalDepth`
"""

from types import MappingProxyType
from typing import FrozenSet, Mapping

from entities.base_entity import Entity


class Animal(Entity[str]):
    """
    Animals in which neurons were recorded.

    Class Attributes
    ----------------
    _OPTIONS : FrozenSet[str]
        Animals represented by their short names (3 letters).
    _naive : FrozenSet[str]
        Naive animals

    Methods
    -------
    :meth:`get_naive`
    :meth:`get_trained`
    """

    _OPTIONS: FrozenSet[str] = frozenset(
        [
            "ath",
            "avo",
            "daf",
            "dai",
            "ele",
            "lem",
            "mor",
            "oni",
            "plu",
            "saf",
            "sir",
            "tan",
            "tel",
            "tul",
            "was",
        ]
    )
    _naive: FrozenSet[str] = frozenset(["mor", "tan"])

    @classmethod
    def get_naive(cls) -> FrozenSet["Animal"]:
        """Naive animals, recorded only during passive listening."""
        return frozenset(cls(o) for o in cls._naive)

    @classmethod
    def get_trained(cls) -> FrozenSet["Animal"]:
        """Trained animals, recorded in both passive and active sessions."""
        return frozenset(cls(o) for o in cls._OPTIONS - cls._naive)

    _LABELS: Mapping[str, str] = MappingProxyType(
        {
            "ath": "Athena",
            "avo": "Avocado",
            "daf": "Daffodil",
            "dai": "Daisy",
            "ele": "Electra",
            "lem": "Lemon",
            "mor": "Morbier",
            "oni": "Onion",
            "plu": "Pluto",
            "saf": "Saffron",
            "sir": "Sirius",
            "tan": "Tango",
            "tel": "Telesto",
            "tul": "Tulip",
            "was": "Wasabi",
        }
    )


class Training(Entity[bool]):
    """
    Training status of an animal.

    Class Attributes
    ----------------
    _OPTIONS : FrozenSet[bool]
        Training statuses represented by boolean values.
        False: Naive animals, recorded only during passive listening.
        True: Trained animals, recorded in both passive and active sessions.
    """

    _OPTIONS: FrozenSet[bool] = frozenset([False, True])
    _LABELS: Mapping[bool, str] = MappingProxyType({True: "Trained", False: "Naive"})


class Area(Entity[str]):
    """
    Brain areas in which neurons were recorded.

    Class Attributes
    ----------------
    _OPTIONS : FrozenSet[str]
        Brain areas represented by their short names (letters).

    Methods
    -------
    :meth:`get_trained`
    :meth:`get_naive`
    """

    _OPTIONS: FrozenSet[str] = frozenset(["A1", "dPEG", "VPr", "PFC"])

    @classmethod
    def get_trained(cls) -> FrozenSet["Area"]:
        """Areas for trained animals (all)."""
        return frozenset(cls(o) for o in cls._OPTIONS)

    @classmethod
    def get_naive(cls) -> FrozenSet["Area"]:
        """Areas for naive animals (all except PFC)."""
        return frozenset(cls(o) for o in cls._OPTIONS - {"PFC"})

    _LABELS: Mapping[str, str] = MappingProxyType(
        {
            "A1": "Primary Auditory Cortex",
            "dPEG": "Dorsal Perigenual Cortex",
            "VPr": "Ventral Prelimbic Cortex",
            "PFC": "Prefrontal Cortex",
        }
    )


class CorticalDepth(Entity):
    """
    Depth of the recording electrode in the cortex.

    Class Attributes
    ----------------
    _OPTIONS : FrozenSet[str]
        Recording depths represented by one letter.
    """

    _OPTIONS: FrozenSet[str] = frozenset(["a", "b", "c", "d", "e", "f"])
