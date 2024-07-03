#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.core_objects.exp_structure` [module]

Classes representing the sequential structure of an experiment ('positional' information).

Classes
-------
:class:`Position`
:class:`Recording`
:class:`Block`
:class:`Slot`
"""

from abc import ABC
from typing import List, Optional


class Position(ABC):
    """
    Any positional information capturing the sequential structure of the experiment.
    
    Class Attributes
    ----------------
    _min: int, optional
        Minimal position.
    _max: int, optional
        Maximal position.

    Attributes
    ----------
    value: int
        Positional information for one event in the experiment.
    
    Methods
    -------
    :meth:`__eq__` : Equal to.
    :meth:`__ne__` : Not equal to.
    :meth:`__lt__` : Less than.
    :meth:`__le__` : Less than or equal to.
    :meth:`__ge__` : Greater than or equal to.
    :meth:`__gt__` : Greater than.
    :meth:`__hash__` : Hash, based on the value.

    Notes
    -----
    Any positional information is represented by integers.
    Positions can be compared to each other based on those integer values.
    The attributes :obj:`_min` and :obj:`_max` (overridden in subclasses)
    define the boundaries of the positional information (if any).

    Implementation
    --------------
    This class does not inherit from CoreObject although it represents an experimental quantity.
    The reason is that representing positional information requires specific logic,
    and thus a dedicated base class.
    This base class is inherited by three concrete sub-classes classes 
    used conjointly to describe the full positional information of an experiment.
    """
    _min: Optional[int] = None
    _max: Optional[int] = None

    def __init__(self, value:int):
        if not isinstance(value, int):
            raise ValueError("Position should be an integer.")
        if self._min is not None and value < self._min:
            raise ValueError(f"Position should be at least {self._min}.")
        if self._max is not None and value > self._max:
            raise ValueError(f"Position should be at most {self._max}.")
        self.value = value

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __ne__(self, other) -> bool:
        return self.value != other.value

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __le__(self, other) -> bool:
        return self.value <= other.value

    def __ge__(self, other) -> bool:
        return self.value >= other.value

    def __gt__(self, other) -> bool:
        return self.value > other.value

    def __hash__(self) -> int:
        return self.value


class Recording(Position):
    """
    Recording number. 

    It corresponds to the order of one session among all the
    sessions performed at one recording site on the same day.
    It is used to order the sessions chronologically.

    Minimal value: 1.    
    Maximal value: None (no upper bound a priori).

    Warning
    -------
    In one site, recording numbers might not start at 1 and may have gaps.
    Indeed, the relevant sessions retained for this study are a subset
    of all the tasks performed in the full data base.
    """
    _min: int = 1

    # No need to override __init__ method.

    def __repr__(self) -> str:
        return f"Recording {self.value} in experiment."


class Block(Position):
    """
    Position of one block of trials within one session.

    Each block of trials corresponds to a sequence of stimuli presentations.
    It includes :
    - At least 1 reference stimulus.
    - At most 7 stimuli in total.
    - At most 1 target stimulus in the sequence, occurring in last position.
    The end of a sequence is usually marked by the occurrence of the target stimulus, 
    except in "catch" trials where it is also a reference stimulus.
    
    The number of blocks presented in a session may vary across sessions.
    Usually, it is of the order of 30-40 blocks.

    Minimal value: 1.    
    Maximal value: None (no fixed bound a priori).

    Warning
    -------
    Block numbers start at 1, not 0 (in contrast to Python indexing)."""
    _min: int = 1

    # No need to override __init__ method.

    def __repr__(self) -> str:
        return f"Block {self.value} in session."


class Slot(Position):
    """
    Slot within one block.

    Each slot marks one period centered around one stimulus presentation.
    It encompasses several epochs:
    - Pre-stimulus baseline
    - Warning period (only in task CLK)
    - Stimulus presentation
    - Post-stimulus period
    - Shock period (always present but effective only in NoGo trials)

    Minimal value: 0
    Maximal value: 7
    """
    _min: int = 0
    _max: int = 7

    # No need to override __init__ method.

    def __repr__(self) -> str:
        return f"Slot {self.value} in block."

    @classmethod
    def get_options(cls) -> List['Slot']:
        """Get all the options for slots in one block."""
        return [cls(i) for i in range(cls._min, cls._max+1)]
