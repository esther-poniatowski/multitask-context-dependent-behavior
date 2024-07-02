#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.core_objects.composites` [module]

Classes representing composite objects storing experimental information.

Those classes cross-reference the core objects and each other.
They can be used as interfaces to recover experimental information from the raw files (TODO).

Classes
-------
:class:`Site`
:class:`Unit`
:class:`Session`
:class:`Trial`

Implementation
--------------
In Sites, Units and Sessions, equality has to be defined based on the ID,
in order to check whether the object is already in a list.
This is necessary to avoid duplicates in bidirectional associations.
"""

from typing import List, Dict, Tuple, Optional

from abc import ABC, abstractmethod

from mtcdb.core_objects.bio import Animal, Area, CorticalDepth, Training
from mtcdb.core_objects.exp_cond import Context, Task, Stimulus
from mtcdb.core_objects.exp_struct import Block, Slot, Recording


class Multiton(ABC):
    """
    Abstract class for composite objects with unique instances based on their IDs.

    Class Attributes
    ----------------
    _instances: Dict[str, 'Multiton']
        Existing instances of the class, indexed by their IDs.

    Attributes
    ----------
    id: str
        Unique identifier of the object.
    initialized: bool
        Whether the object has already been initialized.
        It is necessary 
    
    Methods
    -------
    :meth:`__init__`
    :meth:`__eq__`
    :meth:`__hash__`

    Implementation
    --------------
    Multiton pattern
    - Each instance is unique based on its ID.
    - When a new instance is created, its ID is stored in the class attribute :attr:`_instances`.
    - If an instance with the same ID is requested, the existing instance is returned.
    - Subclasses should implement the :meth:`__init__` method by including the same 
      logic as the abstract template, and setting additional specific attributes.
    """
    _instances: Dict[str, 'Multiton'] = {}

    def __new__(cls, id_:str, *args, **kwargs):
        """
        Create or return an instance of the class.
        
        Parameters
        ----------
        id_: str
            Unique identifier of the object to be created or recovered.
            
        Returns
        -------
        'Multiton'
            Existing instance, if it already exists under the requested ID.
            New instance, if it does not exist yet.
        """
        if id_ in cls._instances:
            return cls._instances[id_]
        else:
            instance = super(Multiton, cls).__new__(cls)
            cls._instances[id_] = instance
            return instance

    @abstractmethod
    def __init__(self, id_:str, *args, **kwargs):
        """
        Initialize the object with its ID, if not already done.
        
        Notes
        -----
        The attribute :attr:`initialized` ensures initialization occurs only once per instance, 
        thus prevents overriding potential changes made after its creation.
        The :meth:`__new__` is not sufficient in itself. 
        Indeed, although an existing instance is *returned* instead of being created, 
        the `__init__` method is called nevertheless after the `__new__` method
        (default behavior when calling a class).
	    The first time an instance is created, :attr:`initialized` does not exist. 
        This allows the initialization code to run and sets :attr:`initialized` to ``True``.
	    On subsequent accesses, the presence of the attribute indicates that the instance
        has already been initialized, so the initialization logic is skipped.
        """
        if not hasattr(self, 'initialized'):
            self.id = id_
            self.initialized = True

    def __eq__(self, other) -> bool:
        """Equality based on IDs."""
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on its IDs."""
        return hash(self.id)


class Site(Multiton):
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
    :class:`Unit`
    :class:`Session`
    """
    def __init__(self, id_:str):
        if not hasattr(self, 'initialized'):
            self.id = id_
            self.initialized = True
            self.units: List['Unit'] = []
            self.sessions: List['Session'] = []

    def __repr__(self) -> str:
        return f"Site {self.id}, {len(self.units)} units, {len(self.sessions)} sessions."

    def recover_units(self, units_metadata) -> List['Unit']:
        """Recover the units recorded at the site."""
        raise NotImplementedError

    def recover_sessions(self, sessions_metadata) -> List['Session']:
        """Recover the sessions performed at the site."""
        raise NotImplementedError

    def add_unit(self, unit:'Unit'):
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

    def add_session(self, session:'Session'):
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


class Unit(Multiton):
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
    def __init__(self, id_:str):
        if not hasattr(self, 'initialized'):
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
        """Representation of the unit."""
        return f"Unit {self.id}, Area {self.area}, Training {self.training}"

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
        s = self.id.split('-')
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


class Session(Multiton):
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
    :class:`Site`
    :class:`mtcdb.core_objects.exp_struct.Recording`
    :class:`mtcdb.core_objects.exp_cond.Task`
    :class:`mtcdb.core_objects.exp_cond.Context`
    """
    def __init__(self, id_:str):
        if not hasattr(self, 'initialized'):
            self.id = id_
            self.initialized = True
            site, rec, ctx, task = self.split_id()
            self.task = Task(task)
            self.ctx = Context(ctx)
            self.set_site(Site(site))
            self.rec = Recording(rec)

    def __repr__(self) -> str:
        """Representation of the session."""
        return f"Session {self.id}, Task {self.task}, Context {self.ctx}"

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
        s = self.id.split('_')
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
    :class:`mtcdb.core_objects.exp_struct.Block`
    :class:`mtcdb.core_objects.exp_struct.Slot`
    :class:`mtcdb.core_objects.exp_cond.Stimulus`
    """
    def __init__(self,
                session:str,
                block:Block,
                slot:Slot
                ):
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
        return f"Slot {self.slot}, Block {self.block}, Session {self.session}, Stimulus {self.stim}"

    def recover_properties(self, events_metadata):
        """Recover the properties of the trial from the metadata."""
        raise NotImplementedError
