#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.bio_info` [module]

Classes representing the biological system under investigation.

Classes
-------
`Animal`
`Training`
`Area`
`CorticalDepth`
"""
from functools import cached_property
import re
from types import MappingProxyType
from typing import FrozenSet, Self, Tuple, TypedDict

from entities.base_entity import Entity


class BrainInfo(str, Entity[str]):
    """
    Brain information in the biological system.

    Define the `__new__` method to inherit from `str`.

    Subclasses should define their own class-level attributes `OPTIONS` and `LABELS` (if applicable)
    and their own methods (in addition to the base `Entity` methods).

    See Also
    --------
    :meth:`Entity.is_valid`
    """

    def __new__(cls, value: str) -> Self:
        if not cls.is_valid(value):  # method from Entity or overridden in subclasses
            raise ValueError(f"Invalid value for {cls.__name__}: {value}.")
        return super().__new__(cls, value)


class Animal(BrainInfo):
    """
    Animals in which neurons were recorded.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Animals represented by their short names (3 letters).
    NAIVE : FrozenSet[str]
        Naive animals, which did not learn any task.

    Attributes
    ----------
    alias : str
        (Core value) Alias of the animal (3 letters).
    is_naive : bool
        (Property) Check if the animal is naive.

    Methods
    -------
    `get_naive`
    `get_trained`
    """

    OPTIONS = frozenset(
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
    NAIVE = frozenset(["mor", "tan"])

    LABELS = MappingProxyType(
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

    @classmethod
    def get_naive(cls) -> FrozenSet[Self]:
        """Naive animals, recorded only during passive listening."""
        return frozenset(cls(o) for o in cls.NAIVE)

    @classmethod
    def get_trained(cls) -> FrozenSet[Self]:
        """Trained animals, recorded in both passive and active sessions."""
        return frozenset(cls(o) for o in cls.OPTIONS - cls.NAIVE)

    @property
    def is_naive(self) -> bool:
        """Check if the animal is naive."""
        return self in self.get_naive()


class Training(int, Entity[bool]):
    """
    Training status of an animal.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[bool]
        Training statuses represented by boolean values.
        False: Naive animals, recorded only during passive listening.
        True: Trained animals, recorded in both passive and active sessions.

    Notes
    -----
    This entity class does not inherit from `BrainInfo` because it is not a string-based
    representation.

    Since it is not possible to inherit from `bool`, the class is defined as a subclass of `int`. It
    implements the special method `__bool__` to provide appropriate behavior in boolean contexts.
    """

    OPTIONS = frozenset([False, True])
    LABELS = MappingProxyType({True: "Trained", False: "Naive"})

    def __new__(cls, value: bool) -> Self:
        return super().__new__(cls, bool(value))

    def __bool__(self) -> bool:
        return bool(int(self))


class Area(BrainInfo):
    """
    Brain areas in which neurons were recorded.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Brain areas represented by their short names (letters).

    Methods
    -------
    `get_trained`
    `get_naive`
    """

    OPTIONS = frozenset(["A1", "dPEG", "VPr", "PFC"])
    LABELS = MappingProxyType(
        {
            "A1": "Primary Auditory Cortex",
            "dPEG": "Dorsal Perigenual Cortex",
            "VPr": "Ventral Prelimbic Cortex",
            "PFC": "Prefrontal Cortex",
        }
    )

    @classmethod
    def get_trained(cls) -> FrozenSet[Self]:
        """Areas for trained animals (all)."""
        return frozenset(cls(o) for o in cls.OPTIONS)

    @classmethod
    def get_naive(cls) -> FrozenSet[Self]:
        """Areas for naive animals (all except PFC)."""
        return frozenset(cls(o) for o in cls.OPTIONS - {"PFC"})


class CorticalDepth(BrainInfo):
    """
    Depth of the recording electrode in the cortex.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Recording depths represented by one letter.
    """

    OPTIONS = frozenset(["a", "b", "c", "d", "e", "f"])


class Site(BrainInfo):
    """
    Recording site, brain location where several sessions were performed.

    Each site contains a set of units (neurons).

    Class Attributes
    ----------------
    ID_PATTERN : re.Pattern
        Regular expression pattern to match a valid site ID.
    EXCEPTION_PATTERN : re.Pattern
        Regular expression pattern to match a valid site ID with an exception (full animal name).
    SITE_PATTERN : re.Pattern
        Regular expression pattern to match a valid site ID in a string, used by other classes in
        hierarchical patterns containing site IDs. Note: It is not used within the `Site` class
        itself since it does not provide fined-grained specification of the site's components.
    DEFAULT_VALUE : str
        Default value for a missing component in the site ID.

    Attributes
    ----------
    id : str
        (Core value) Identifier of the site. Example: ``'avo052a'``
    animal : Animal
        (Property) Animal in which the site was recorded (see `Animal`).
    depth : CorticalDepth
        (Property) Depth of the site in the cortex (see `CorticalDepth`).

    Methods
    -------
    `is_valid` (override the method from the base class `Entity`)
    `split_id`

    Notes
    -----
    This class does not define any class-level attributes `OPTIONS` nor `LABELS` because the valid
    values are based on a pattern match rather than on a predefined set of labels.
    Therefore, the `is_valid` class method is overridden to enforce the pattern match.
    """

    ID_PATTERN = re.compile(r"^(?P<animal>[a-z]{3})(?P<tag>[0-9]{3})(?P<depth>[a-z])$")
    EXCEPTION_PATTERN = re.compile(r"^(?P<animal>lemon)(?P<tag>[0-9]{3})(?P<depth>[a-z])$")
    SITE_PATTERN = r"[a-z]{3,5}[0-9]{3}[a-z]"
    DEFAULT_VALUE = ""

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """
        Check if the value is a valid site ID. Override the method from the base class `Entity`.

        Example
        -------
        >>> Site.is_valid("avo052a")
        True

        >>> Site.is_valid("lemon052a")
        True

        See Also
        --------
        :meth:`split_id`
        :meth:`Animal.is_valid`
        :meth:`CorticalDepth.is_valid`
        """
        animal, rec, depth = cls.split_id(value)
        if not all([animal, rec, depth]):  # checks for any empty string
            return False
        return Animal.is_valid(animal) and CorticalDepth.is_valid(depth)

    @classmethod
    def split_id(cls, value: str) -> Tuple[str, str, str]:
        """
        Split a site's ID into its components.

        Format of the site's ID (regular expression): ``{animal}{rec}{depth}``

        - {animal} [3 letters] Alias of one animal, as defined in `Animal` (e.g. ``'avo'``)
        - {tag}    [3 digits ] Numerical tag to identify this site in the animal (e.g. ``'052'``)
        - {depth}  [1 letter ] Depth in the cortex, as defined in `CorticalDepth` (e.g. ``'a'``).

        Exception: 'lemon' (full animal name) is allowed in the string instead of its alias 'lem'.
        In this case, only the first 3 characters are kept for the animal's ID.

        In total, the length of the site's ID is at least 7 characters, and at most 9 characters for
        the exception case.

        Example
        -------
        >>> site = Site("avo052a")
        >>> site.split_id()
        ('avo', '052', 'a')

        >>> site = Site("lemon052a")
        >>> site.split_id()
        ('lem', '052', 'a')

        Returns
        -------
        animal, tag, depth : Tuple[str, str, str]
            Animal, tag, depth components of the site's ID.
            If the ID is invalid, returns empty strings for all components (for type consistency).

        See Also
        --------
        :func:`re.match`
            Input: regular expression pattern and string to match.
            Output: match object or None if no match is found.
        :meth:`re.Match.group`
            Extract the matched groups defined in the regular expression pattern by parentheses.
        """
        match = cls.ID_PATTERN.match(value) or cls.EXCEPTION_PATTERN.match(value)
        if not match:  # default values
            animal, tag, depth = (cls.DEFAULT_VALUE,) * 3
        else:  # extract components
            animal = match.group("animal")[:3]  # exception (full name): keep only 3 first letters
            tag, depth = match.group("tag"), match.group("depth")
        return animal, tag, depth

    class SiteComponents(TypedDict):
        """Typed dictionary for the components of a site's ID."""

        animal: Animal
        tag: int
        depth: CorticalDepth

    @cached_property
    def id_components(self) -> SiteComponents:
        """
        Caches the components of the site in a dictionary for easy access, in appropriate types.

        Returns
        -------
        SiteComponents
        """
        animal_str, tag_str, depth_str = self.split_id(self)
        animal = Animal(animal_str)
        tag = int(tag_str)
        depth = CorticalDepth(depth_str)
        return {"animal": animal, "tag": tag, "depth": depth}

    @property
    def animal(self) -> Animal:
        """
        Get the animal where the site was recorded.
        """
        return self.id_components["animal"]

    @property
    def depth(self) -> CorticalDepth:
        """
        Get the depth of the site in the cortex.
        """
        return self.id_components["depth"]


class Unit(BrainInfo):
    """
    Single unit (neuron).

    Class Attributes
    ----------------
    ID_PATTERN : re.Pattern
        Regular expression pattern to match a valid unit ID.
    DEFAULT_VALUE : str
        Default value for a missing component in the unit ID.

    Attributes
    ----------
    id : str
        (Core value) Identifier of the unit. Example: ``'avo052a-d1'``
    site : Site
       (Property) Site where the unit was recorded. Example : ``'avo052a'``
    animal : Animal
        (Property) Animal where the unit was recorded. Example : ``'avo'``
    depth : CorticalDepth
        (Property) Depth of the unit along the electrode. Example : ``'a'``
    electrode : str
        (Property) Electrode channel where the unit was recorded. Example : ``'d'``

    Methods
    -------
    `is_valid` (override the method from the base class `Entity`)
    `split_id`
    """

    ID_PATTERN = re.compile(f"^(?P<site>{Site.SITE_PATTERN})-(?P<el>[a-z])(?P<u>[0-9])$")
    DEFAULT_VALUE = ""

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """
        Check if the value is a valid unit ID. Override the method from the base class `Entity`.

        Example
        -------
        >>> Unit.is_valid("avo052a-d1")
        True

        See Also
        --------
        :meth:`Site.is_valid`
        """
        site_id, electrode, unit_num = cls.split_id(value)
        if not all([site_id, electrode, unit_num]):  # checks for any empty string
            return False
        return Site.is_valid(site_id)

    @classmethod
    def split_id(cls, value: str) -> Tuple[str, str, str]:
        """
        Split a unit's ID into its components.

        Format of the unit's ID (regular expression): ``{site}-{el}{u}`` (e.g. ``'avo052a-d1'``)

        - {site} [...] Identifier for the site, as defined in `Site.is_valid` (e.g. ``'avo052a'``)
        - {el}   [1 letter] Electrode channel (e.g. ``'d'``)
        - {u}    [1 digit ] Unit index relative to this electrode (e.g. ``'1'``)

        Example
        -------
        >>> unit = Unit("avo052a-d1")
        >>> unit.split_id()
        ('avo052a', 'd', '1')

        Returns
        -------
        site_id, electrode, unit_num : Tuple[str, str, str]
            Site, animal, electrode components of the unit's ID.
            If the ID is invalid, returns empty strings for all components (for type consistency).
        """

        match = cls.ID_PATTERN.match(value)
        if not match:
            site_id, electrode, unit_num = (cls.DEFAULT_VALUE,) * 3
        else:
            site_id, electrode, unit_num = match.group("site"), match.group("el"), match.group("u")
        return site_id, electrode, unit_num

    class UnitComponents(TypedDict):
        """Typed dictionary for the components of a unit's ID."""

        site: Site
        electrode: str
        unit_num: int

    @cached_property
    def id_components(self) -> UnitComponents:
        """
        Caches the components of the unit in a dictionary for easy access, in appropriate types.

        Returns
        -------
        UnitComponents
        """
        site_str, el_str, unit_str = self.split_id(self)
        site = Site(site_str)
        unit = int(unit_str)
        return {"site": site, "electrode": el_str, "unit_num": unit}

    @property
    def site(self) -> Site:
        """
        Get the site where the unit was recorded.
        """
        return self.id_components["site"]

    @property
    def animal(self) -> Animal:
        """
        Get the animal where the unit was recorded (delegate to `site`).
        """
        return self.site.animal

    @property
    def depth(self) -> CorticalDepth:
        """
        Get the depth of the unit along the electrode (delegate to `site`).
        """
        return self.site.depth

    @property
    def electrode(self) -> str:
        """
        Get the electrode channel where the unit was recorded.
        """
        return self.id_components["electrode"]
