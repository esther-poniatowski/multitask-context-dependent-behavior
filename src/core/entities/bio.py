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

from entities.base_entity import CoreObject


class Animal(CoreObject[str]):
    """
    Animals in which neurons were recorded.

    Class Attributes
    ----------------
    _options : FrozenSet[str]
        Animals represented by their short names (3 letters).
    _naive : FrozenSet[str]
        Naive animals

    Methods
    -------
    :meth:`get_naive`
    :meth:`get_trained`
    """

    _options: FrozenSet[str] = frozenset(
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
        return frozenset(cls(o) for o in cls._options - cls._naive)

    _full_labels: Mapping[str, str] = MappingProxyType(
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


class Training(CoreObject[bool]):
    """
    Training status of an animal.

    Class Attributes
    ----------------
    _options : FrozenSet[bool]
        Training statuses represented by boolean values.
        False: Naive animals, recorded only during passive listening.
        True: Trained animals, recorded in both passive and active sessions.
    """

    _options: FrozenSet[bool] = frozenset([False, True])
    _full_labels: Mapping[bool, str] = MappingProxyType({True: "Trained", False: "Naive"})


class Area(CoreObject[str]):
    """
    Brain areas in which neurons were recorded.

    Class Attributes
    ----------------
    _options : FrozenSet[str]
        Brain areas represented by their short names (letters).

    Methods
    -------
    :meth:`get_trained`
    :meth:`get_naive`
    """

    _options: FrozenSet[str] = frozenset(["A1", "dPEG", "VPr", "PFC"])

    @classmethod
    def get_trained(cls) -> FrozenSet["Area"]:
        """Areas for trained animals (all)."""
        return frozenset(cls(o) for o in cls._options)

    @classmethod
    def get_naive(cls) -> FrozenSet["Area"]:
        """Areas for naive animals (all except PFC)."""
        return frozenset(cls(o) for o in cls._options - {"PFC"})

    _full_labels: Mapping[str, str] = MappingProxyType(
        {
            "A1": "Primary Auditory Cortex",
            "dPEG": "Dorsal Perigenual Cortex",
            "VPr": "Ventral Prelimbic Cortex",
            "PFC": "Prefrontal Cortex",
        }
    )


class CorticalDepth(CoreObject):
    """
    Depth of the recording electrode in the cortex.

    Class Attributes
    ----------------
    _options : FrozenSet[str]
        Recording depths represented by one letter.
    """

    _options: FrozenSet[str] = frozenset(["a", "b", "c", "d", "e", "f"])
