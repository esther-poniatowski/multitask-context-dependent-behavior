#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.core_obj.base_enum` [module]

Define an abstract base class for all enumeration classes.

Classes
-------
:class:`CoreObject` (Enum, ABC)

Implementation
--------------
The base class defines a common interface for all classes representing 
the core objects in the package. It should be inherited by subclasses
that represent specific types of objects, which correspond to concrete 
quantities or categories in the experiment.
Each subclass stands as a type in itself and provides a central documentation
for the object it represents. It stores :
- A set of valid values for the object it represents.
- Metadata (optional) about the type of object, often used to filter 
  the valid objects based on various criteria.

See Also
--------
:class:`abc.ABC`
"""

from abc import ABC
from types import MappingProxyType
from typing import Generic, TypeVar, Mapping, FrozenSet


T = TypeVar("T")
"""Type variable for the values representing a core object."""


class CoreObject(ABC, Generic[T]):
    """
    Abstract base class for custom enumeration classes.
    
    Class Attributes
    ----------------
    _options: FrozenSet[T]
        Valid values to represent the object.
        To be overridden in the derived classes.
    _full_labels: Mapping[T, str]
        Full labels for the valid values.
        Keys: Valid values.
        Values: Full labels for the valid values.
        To be overridden in the derived classes.

    Methods
    -------
    :meth:`__init__`
    :attr:`full_label`
    :meth:`get_options`
    :meth:`get_labels`
    :meth:`__eq__`
    :meth:`__hash__`

    Examples
    --------
    Assuming a subclass :class:`ConcreteObject` with valid values "a" and "b":
    .. code-block:: python
        class ConcreteObject(CoreObject):
            _options = ["a", "b"]
            _full_labels = {"a": "Alpha", "b": "Beta"}
    
    Instantiate an object and get its full label:
    .. code-block:: python
        obj = ConcreteObject("a")
        print(obj.full_label)  # "Alpha"
    
    Check if two objects are equal:
    .. code-block:: python
        obj1 = ConcreteObject("a")
        obj2 = ConcreteObject("a")
        obj3 = ConcreteObject("b")
        print(obj1 == obj2)  # True
        print(obj1 == obj3)  # False
    
    Check if an object belongs to a list:
    .. code-block:: python
        items = [ConcreteObject("a"), ConcreteObject("a")]
        print(obj1 in items)  # True
        print(obj3 in items)  # False
    """
    _options: FrozenSet[T] = frozenset()
    _full_labels: Mapping[T, str] = MappingProxyType({})

    def __init__(self, value):
        if value not in self._options:
            raise ValueError(f"Invalid value: {value}")
        self.value = value

    @property
    def full_label(self) -> str:
        """Full label for the value (often used for visualization)."""
        return self._full_labels.get(self.value, "")

    @classmethod
    def get_options(cls) -> FrozenSet[T]:
        """Get all the valid options for the values of an object."""
        return cls._options # type: ignore

    def __eq__(self, other) -> bool:
        """Check equality based on the value."""
        if isinstance(other, CoreObject):
            return self.value == other.value
        return False

    def __hash__(self) -> int:
        """Provide a hash based on the value.
        
        Usages :
        - Set the object as a key in a dictionary.
        - Include the object in a set.
        - Check containment: 
          The hash value is used to quickly locate the possible match in the collection, 
          and then the method :meth:`__eq__` is used to verify equality.
        
        If two objects are considered equal (via :meth:`__eq__`), 
        they must have the same hash value to ensures that they will be treated
        as the same object in hash-based collections.
        Therefore, the hash value should be based on the value of the object.
        """
        return hash(self.value)
