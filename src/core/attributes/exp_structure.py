#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.attributes.exp_structure` [module]

Classes representing the sequential structure of an experiment ('positional' information).

Classes
-------
ExpStructure
Recording
Block
Slot
Session
"""
from functools import cached_property
import re
from typing import Optional, Self, Tuple, List

from core.attributes.base_attribute import Attribute
from core.attributes.exp_factors import Task, Attention
from core.attributes.brain_info import Site


class ExpStructure(int, Attribute[int]):
    """
    Any positional information capturing the sequential structure of the experiment.

    Define the `__new__` method to inherit from `int`.

    Subclasses should define their own class-level attributes `MIN` and `MAX` (if applicable) and
    their own methods (in addition to the base `Attribute` methods).

    Class Attributes
    ----------------
    MIN, MAX : Optional[int]
        Minimal and maximal values for the positional information (overridden in subclasses). They
        define the boundaries of the positional information (if any).

    Methods
    -------
    is_valid (override the method from the base class `Attribute`)

    Notes
    -----
    By default, the position base class does not define the `OPTIONS` attribute, as the position
    might not be bounded. However, it is possible to define it in subclasses if needed.
    """

    MIN: Optional[int] = None
    MAX: Optional[int] = None

    def __new__(cls, value: int) -> Self:
        if cls.is_valid(value):  # method from the current subclass
            raise ValueError(
                f"Invalid value for {cls.__name__}: {value} out of bounds "
                f"(min: {cls.MIN}, max: {cls.MAX})."
            )
        return super().__new__(cls, value)

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """
        Check if the value is a valid position, within the defined boundaries.

        Override the method from the base class `Attribute`.
        """
        if cls.MIN is not None and value < cls.MIN:
            return False
        if cls.MAX is not None and value > cls.MAX:
            return False
        return True


class Recording(ExpStructure):
    """
    Recording number for one session in the experiment at a given site.

    It corresponds to the order of one session among all the sessions performed at one recording
    site on the same day. It is used to order the sessions chronologically.

    Minimal value: 1.
    Maximal value: None (no upper bound a priori).

    Warning
    -------
    In one site, recording numbers might not start at 1 and may have gaps. Indeed, the relevant
    sessions retained for this study are a subset of all the tasks performed in the full data base.
    """

    MIN = 1


class Block(ExpStructure):
    """
    ExpStructure of one block of trials within one session.

    Each block of trials corresponds to a sequence of stimuli presentations. It includes :

    - At least 1 reference stimulus.
    - At most 7 stimuli in total.
    - At most 1 target stimulus in the sequence, occurring in last position.

    The end of a sequence is usually marked by the occurrence of the target stimulus, except in
    "catch" trials where it is also a reference stimulus.

    The number of blocks presented in a session may vary across sessions.
    Usually, it is of the order of 30-40 blocks.

    Minimal value: 1.
    Maximal value: None (no fixed bound a priori).

    Warning
    -------
    Block numbers start at 1, not 0 (in contrast to Python indexing).
    """

    MIN = 1


class Slot(ExpStructure):
    """
    Slot within one block.

    Each slot marks one period centered around one stimulus presentation.

    Each slot encompasses several epochs:

    - Pre-stimulus baseline
    - Warning period (only in task CLK)
    - Stimulus presentation
    - Post-stimulus period
    - Shock period (always present but effective only in NoGo trials)

    Minimal value: 0
    Maximal value: 7
    """

    MIN = 0
    MAX = 7
    OPTIONS = frozenset(range(MIN, MAX + 1))


class Session(str, Attribute[str]):
    """
    Recording session, composed of a set of trials.

    Define the `__new__` method to inherit from `str`.

    Class Attributes
    ----------------
    ID_PATTERN : re.Pattern
        Regular expression pattern to match a valid session ID.
    DEFAULT_VALUE : str
        Default value for a missing component in the session ID.

    Arguments
    ---------
    value : str
        (Core value) Identifier of the session. Example: ``'avo052a04_p_PTD'``

    Attributes
    ----------
    site : Site
        Location where the session was recorded. Example : ``'avo052a'``
    recording : Recording
        Recording number associated to the session at this site (used for ordering sessions).
    task : Task
        Task performed during the session.
    attention : Attention
        Engagement of the animal in the task.

    Methods
    -------
    is_valid (override the method from the base class `Attribute`)
    split_id
    order (class method)

    See Also
    --------
    `core.attributes.brain_info.Site`
    `core.attributes.exp_structure.Recording`
    `core.attributes.exp_factors.Task`
    `core.attributes.exp_factors.Attention`
    """

    ID_PATTERN = re.compile(
        f"^(?P<site>{Site.SITE_PATTERN})(?P<rec>[0-9]{2})_(?P<attn>[a-z])_(?P<task>[A-Z]{3})$"
    )
    DEFAULT_VALUE = ""

    def __new__(cls, value: str) -> Self:
        if not cls.is_valid(value):  # overridden in this subclass
            raise ValueError(f"Invalid value for {cls.__name__}: {value}.")
        return super().__new__(cls, value)

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """
        Check if the value is a valid session ID. Override the method from the base class
        `Attribute`.

        Examples
        --------
        >>> Session.is_valid("avo052a04_p_PTD")
        True

        See Also
        --------
        `Site.is_valid`
        `Recording.is_valid` (method from the class `ExpStructure`)
        `Attention.is_valid` (method from the class `ExpFactor` inheriting from `Attribute`)
        `Task.is_valid`      (idem)
        """
        site, rec, attn, task = cls.split_id(value)
        if not all([site, rec, attn, task]):  # checks for any empty string or zero
            return False
        return (
            Site.is_valid(site)
            and Recording.is_valid(int(rec))  # convert string to int first
            and Attention.is_valid(attn)
            and Task.is_valid(task)
        )

    @classmethod
    def split_id(cls, value) -> Tuple[str, str, str, str]:
        """
        Split the session ID into its components.

        Format of the session's ID: ``{site}{rec}_{attn}_{task}`` (e.g. ``'avo052a04_p_PTD'``).

        - {site} [...] Identifier for the site, as defined in `Site.is_valid` (e.g. ``'avo052a'``)
        - {rec}  [2 digits] Recording number at this site (e.g. ``'04'``)
        - {attn}  [1 letter] Attention of the session, passive or active (``'p'`` or ``'a'``)
        - {task} [3 letters] Task performed during the session (``'PTD'``, ``'CLK'``, ``'CCH'``)

        Returns
        -------
        Tuple[str, str, str, str]
            Site, recording number, attentional state, task.

        Example
        -------
        >>> session = Session("avo052a04_p_PTD")
        >>> session.split_id()
        ('avo052a', '04', 'p', 'PTD')
        """
        match = cls.ID_PATTERN.match(value)
        if not match:
            return (cls.DEFAULT_VALUE,) * 4
        return match.group("site"), match.group("rec"), match.group("attn"), match.group("task")

    @cached_property
    def site(self) -> Site:
        """
        Brain site at which the session was recorded (cached).
        """
        site_str, _, _, _ = self.split_id(self)
        return Site(site_str)

    @cached_property
    def recording(self) -> Recording:
        """
        Recording number of the session at the site (cached).
        """
        _, rec_str, _, _ = self.split_id(self)
        return Recording(int(rec_str))  # convert string to int first

    @cached_property
    def attention(self) -> Attention:
        """
        Attentional state of the animal in the session (cached).
        """
        _, _, attn_str, _ = self.split_id(self)
        return Attention(attn_str)

    @cached_property
    def task(self) -> Task:
        """
        Task performed during the session (cached).
        """
        _, _, _, task_str = self.split_id(self)
        return Task(task_str)

    @classmethod
    def order(cls, *args: Self) -> List[Self]:
        """
        Order sessions chronologically, based on their recording number.

        Parameters
        ----------
        args : Session
            Sessions to order.

        Returns
        -------
        List[Session]
            Ordered list of sessions.

        Examples
        --------
        >>> s1 = Session("avo052a04_p_PTD")
        >>> s2 = Session("avo052a05_p_PTD")
        >>> s3 = Session("avo052a03_p_PTD")
        >>> Session.order(s1, s2, s3)
        [s3, s1, s2]

        See Also
        --------
        `sorted`: Python built-in function to sort iterables based on a key function.
        """
        return sorted(args, key=lambda session: session.recording)
