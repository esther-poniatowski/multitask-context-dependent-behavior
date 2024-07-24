#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.core_objects.composites` [module]

Classes representing composite objects storing experimental information.

Those classes cross-reference the core objects and each other.
They can be used as interfaces to recover experimental information from the raw files (TODO).

Classes
-------
:class:`Multiton`
:class:`Site`
:class:`Unit`
:class:`Session`
:class:`Trial`
"""

from typing import List, Dict, Tuple, Optional, Any, Type

from mtcdb.core_objects.bio import Animal, Area, CorticalDepth, Training
from mtcdb.core_objects.exp_condition import Context, Task, Stimulus
from mtcdb.core_objects.exp_structure import Block, Slot, Recording


class Multiton(type):
    """
    Metaclass for composite objects with unique instances based on their IDs.

    Class Attributes
    ----------------
    _instances:
        Existing instances of the class, indexed by their IDs.
        Repertoire of classes using this metaclass and their respective instances.
        Keys: Classes that use :class:`Multiton` as their metaclass.
        Values: Dictionaries that map unique :attr:`id` to instances of the respective class.

    Attributes
    ----------
    __eq__: Callable
        Equality method based on the ID.
    __hash__: Callable
        Hash method based on the ID.

    Methods
    -------
    :meth:`__call__`
    :meth:`__init__`

    Warning
    -------
    In the context of a metaclass, classes (cls) are considered as instances.

    Implementation
    --------------
    The metaclass implements the Multiton Pattern.
    For each class using the metaclass:
    - Each instance is unique based on its ID.
    - When a new instance is created, its ID is stored in the class attribute :attr:`_instances`.
    - If an instance with the same ID is requested, the existing instance is returned.

    The metaclass also defines equality and hash methods to be consistent
    with the uniqueness of the instances based on their IDs.
    This is necessary to avoid duplicates in bidirectional associations.
    """

    _instances: Dict[Type, Dict[Any, Any]] = {}

    def __call__(cls, id_, *args, **kwargs) -> Any:
        """
        Create a new instance or return an existing instance of the class.

        Parameters
        ----------
        cls: type
            Class being instantiated.
        id_: str
            Unique identifier of the instance.

        Returns
        -------
        Any:
            Instance of the class with the specified :attr:`id` (new or existing).

        Notes
        -----
        This method is called each time the class is *instantiated*,
        (i.e. when the class is *called* as a function).

        Implementation
        --------------
        Each class using the metaclass has its own instance repertoire within :attr:`_instances`.
        This allows each class to maintain its own set of instances,
        while sharing the instance management logic provided by the metaclass.
        """
        if cls not in cls._instances:
            cls._instances[cls] = {}
        if id_ not in cls._instances[cls]:
            instance = super().__call__(id_, *args, **kwargs)
            cls._instances[cls][id_] = instance
        return cls._instances[cls][id_]

    def __init__(cls, name, bases, dct):
        """
        Define instance methods :meth:`__eq__` and :meth:`__hash__` for the class.

        Parameters
        ----------
        cls: type
            Class being created.
        name: str
            Name of the class.
        bases: Tuple[type]
            Base classes of the class.
        dct: Dict[str, Any]
            Class dictionary containing its attributes and methods.

        Notes
        -----
        This method is called when a class using the metaclass is *defined*
        (i.e. instantiated, as an *instance* of the metaclass).
        """
        # Initialize the class as an instance of the metaclass
        super().__init__(name, bases, dct)

        def __eq__(self, other) -> bool:
            """Equality based on IDs."""
            return self.id == other.id

        def __hash__(self) -> int:
            """Hash based on its IDs."""
            return hash(self.id)

        # Attach instance methods to the class
        cls.__eq__ = __eq__  # type: ignore
        cls.__hash__ = __hash__  # type: ignore


class Site(metaclass=Multiton):
    """
    Recording site, location where several sessions were performed.

    Each site leads to a set of units (neurons)
    and a set of sessions (recordings).
    It is used to assess artifacts in units' activity at the same site.

    Attributes
    ----------
    id: str
        Identifier of the site.
        Example: ``'avo052a'``
    units: List['Unit']
        All the units recorded at this site.
    sessions: List['Session']
        All the sessions performed at this site.

    Methods
    -------
    :meth:`recover_units`
    :meth:`recover_sessions`
    :meth:`add_unit`
    :meth:`add_session`

    See Also
    --------
    :class:`Multiton`
    :class:`Unit`
    :class:`Session`
    """

    def __init__(self, id_: str):
        self.id = id_
        self.initialized = True
        self.units: List["Unit"] = []
        self.sessions: List["Session"] = []

    def __repr__(self) -> str:
        return f"Site {self.id}: {len(self.units)} units, {len(self.sessions)} sessions."

    def recover_units(self, units_metadata) -> List["Unit"]:
        """Recover the units recorded at the site."""
        raise NotImplementedError

    def recover_sessions(self, sessions_metadata) -> List["Session"]:
        """Recover the sessions performed at the site."""
        raise NotImplementedError

    def add_unit(self, unit: "Unit"):
        """
        Add a unit to the site.

        Implementation
        --------------
        Bidirectional association pattern
        - To ensure the consistency of the relationship, when a unit is added to a site,
        the site of the unit should also be set if it is not already done.
        - To avoid infinite recursion, the first condition checks
        whether the unit is already recorded in the site.

        See Also
        --------
        :meth:`Unit.set_site`
        """
        if unit not in self.units:
            self.units.append(unit)
            try:
                unit.set_site(self)
            except AttributeError:
                pass

    def add_session(self, session: "Session"):
        """
        Add a session to the site.

        Implementation
        --------------
        This function is part of a bidirectional association pattern.

        See Also
        --------
        :meth:`Session.set_site`
        :meth:`add_unit` for similar implementation.
        """
        if session not in self.sessions:
            self.sessions.append(session)
            try:
                session.site = self
            except AttributeError:
                pass


class Unit(metaclass=Multiton):
    """
    Single unit (neuron).

    Attributes
    ----------
    id: str
        Identifier of the unit.
        Example: ``'avo052a-d1'``
    site: Site
        Location where the unit was recorded.
        Example : ``'avo052a'``
    area: Area
        Brain area where the unit was recorded.
    depth: CorticalDepth
        Depth of the unit along the electrode.
    training: Training
        Whether the unit was recorded in a trained animal.
    animal: Animal
        Animal identifier.
    sessions: List['Session']
        All the sessions in which the unit was recorded.

    Methods
    -------
    :meth:`split_id`
    :meth:`recover_properties`
    :meth:`recover_sessions`
    :meth:`set_site`

    See Also
    --------
    :class:`Multiton`
    :class:`Site`
    :class:`Session`
    :class:`mtcdb.core_objects.bio.Animal`
    :class:`mtcdb.core_objects.bio.Area`
    :class:`mtcdb.core_objects.bio.CorticalDepth`
    :class:`mtcdb.core_objects.bio.Training`

    Notes
    -----
    Most attributes are set automatically without the need of passing parameters.
    """

    def __init__(self, id_: str):
        self.id = id_
        self.initialized = True
        site, animal, depth = self.split_id()
        self.set_site(Site(site))
        self.animal = Animal(animal)
        self.depth = CorticalDepth(depth)
        self.training = Training(self.animal in Animal.get_trained())
        self.area: Optional[Area] = None
        self.sessions: List[Session] = []

    def __repr__(self):
        return f"Unit {self.id}: Area {self.area}, Training {self.training}"

    def split_id(self) -> Tuple[str, str, str]:
        """
        Split the unit ID into its components.

        Format of the unit's ID :
        ``{site}-{el}{u}``
        with {site} = {animal}{rec}{depth}
        Example : ``'avo052a-d1'``
        - {site} : 3 letters (e.g. ``'avo'``),
                   3 numbers (e.g. ``'052'``),
                   1 letter (e.g. ``'a'``).
        - {depth}: last letter of {site} (e.g. ``'a'``),
        - {el}   : 1 letter (e.g. ``'d'``),
                    electrode channel
        - {u}    : 1 number (e.g. ``'1'``),
                    unit number attributed by the spike sorting algorithm.

        Example
        -------
        >>> unit = Unit("avo052a-d1")
        >>> unit.split_id()
        ('avo052a', 'avo', 'a')
        """
        s = self.id.split("-")
        site = s[0]
        animal = s[0][:3]
        depth = s[0][-1]
        return site, animal, depth

    def recover_properties(self, units_metadata):
        """Recover the properties of the unit from the metadata."""
        raise NotImplementedError

    def recover_sessions(self, sessions_metadata):
        """Recover the sessions in which the unit was recorded."""
        raise NotImplementedError

    def set_site(self, site: Site):
        """
        Set the site of the unit.

        Implementation
        --------------
        Bidirectional association pattern
        - To ensure the consistency of the relationship, when a site is set to a unit,
        the unit should be added to the site's units list if it is not already done.
        - To avoid infinite recursion, the first condition checks
        whether the unit is already recorded in the site.

        See Also
        --------
        :meth:`Site.add_unit`
        """
        if self not in site.units:
            site.units.append(self)
            self.site = site


class Session(metaclass=Multiton):
    """
    Recording session, composed of a set of trials.

    Attributes
    ----------
    id: str
        Identifier of the session.
        Example: ``'avo052a04_p_PTD'``
    site: str
        Location where the session was recorded.
        Example : ``'avo052a'``
    rec: int
        Recording number at this site (used for ordering the sessions).
    task: Task
        Task performed during the session.
    ctx: Context
        Engagement of the animal in the task.

    Methods
    -------
    :meth:`split_id`
    :meth:`set_site`

    See Also
    --------
    :class:`Multiton`
    :class:`Site`
    :class:`mtcdb.core_objects.exp_structure.Recording`
    :class:`mtcdb.core_objects.exp_condition.Task`
    :class:`mtcdb.core_objects.exp_condition.Context`
    """

    def __init__(self, id_: str):
        self.id = id_
        self.initialized = True
        site, rec, ctx, task = self.split_id()
        self.task = Task(task)
        self.ctx = Context(ctx)
        self.set_site(Site(site))
        self.rec = Recording(rec)

    def __repr__(self) -> str:
        return f"Session {self.id}: Task {self.task}, Context {self.ctx}"

    def split_id(self) -> Tuple[str, int, str, str]:
        """
        Split the session ID into its components.

        Format of the session's ID :
        ``{site}{rec}_{ctx}_{task}``
        Example : ``'avo052a04_p_PTD'``
        - {site} : 3 letters (e.g. ``'avo'``),
                   3 numbers (e.g. ``'052'``),
                   1 letter (e.g. ``'a'``).
        - {rec}  : 2 numbers (e.g. ``'04'``)
        - {ctx}  : 1 letter (e.g. ``'p'``)
        - {task} : 3 letters (e.g. ``'PTD'``)

        Example
        -------
        >>> session = Session("avo052a04_p_PTD")
        >>> session.split_id()
        ('avo052a', 4, 'p', 'PTD')

        Notes
        -----
        For the animal Lemon, the site uses the full name 'lemon'.
        Therefore, Sites' IDs should be obtained FROM THE END of the string:
        site[:-1] instead of site[:5] (which would not be valid for 'lemon').
        """
        s = self.id.split("_")
        site = s[0][:-2]
        rec = int(s[0][-2:])
        ctx = s[1]
        task = s[2]
        return site, rec, ctx, task

    def set_site(self, site: Site):
        """
        Set the site of the session.

        Implementation
        --------------
        This function is part of a bidirectional association pattern.

        See Also
        --------
        :meth:`Site.add_session`
        """
        if self not in site.sessions:
            site.sessions.append(self)
            self.site = site


class Trial:
    """
    Single trial, centered on one stimulus presentation.

    Notes
    -----
    This object is mainly used for data pre-processing,
    where trials's structure still differ from each other.
    Afterwards, in the majority of the analyses,
    trials's epochs are uniformly and their properties are
    directly stored in the data coordinates.

    Attributes
    ----------
    session: str
        ID of the session to which the trial belongs.
    rec: Recording
        Recording number of the session.
    block: Block
        Block within the session to which the trial belongs.
    slot: Slot
        Slot in the block to which the trial belongs.
    stim: Stimulus
        Stimulus presented to the animal.
    t_start: float
        Start time within the block (in seconds).
    t_end: float
        End time within the block (in seconds).
    t_warn: float, optional
        Warning onset time within the block (in seconds).
        Only in task CLK.
    t_on: float
        Stimulus onset time within the block (in seconds).
    t_off: float
        Stimulus offset time within the block (in seconds).
    t_shock: float, optional
        Shock time within the block (in seconds).
        Only in error trials.
    error: bool, optional
        Whether the animal behaved incorrectly in the response period.
        Only for target stimuli.

    Methods
    -------
    :meth:`recover_properties`

    See Also
    --------
    :class:`Session`
    :class:`mtcdb.core_objects.exp_structure.Block`
    :class:`mtcdb.core_objects.exp_structure.Slot`
    :class:`mtcdb.core_objects.exp_condition.Stimulus`
    """

    def __init__(self, session: str, block: Block, slot: Slot):
        self.session: str = session
        self.block: Block = block
        self.slot: Slot = slot
        self.stim: Optional[Stimulus] = None
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.t_on: Optional[float] = None
        self.t_off: Optional[float] = None
        self.t_warn: Optional[float] = None
        self.t_shock: Optional[float] = None
        self.error: Optional[bool] = None

    def __repr__(self) -> str:
        return f"Trial: Slot {self.slot}, Block {self.block}, Session {self.session}, Stimulus {self.stim}"

    def recover_properties(self, events_metadata):
        """Recover the properties of the trial from the metadata."""
        raise NotImplementedError
