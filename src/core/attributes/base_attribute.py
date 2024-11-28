#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.attributes.base_attribute` [module]

Define a common interface for all classes representing attributes (core objects) in the package.

Classes
-------
Attribute (generic, base class)
"""
from typing import (
    Generic,
    TypeVar,
    Mapping,
    FrozenSet,
    Any,
    cast,
    Iterable,
    Union,
    List,
    Tuple,
    Type,
    Set,
)


BaseT = TypeVar("BaseT", int, str, float, bool)
"""Type variable for the basic type from which the attribute inherits."""


class Attribute(Generic[BaseT]):
    """
    Mixin class for attribute types, providing common validation and labeling functionality.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[BaseT], default=frozenset()
        Valid values for the attribute, if applicable. To be overridden in derived classes.
    LABELS : Mapping[BaseT, str], default=MappingProxyType({})
        Full labels for the valid values, if applicable. To be overridden in derived classes.
        Keys: Valid values.
        Values: Full labels for the valid values.

    Methods
    -------
    is_valid
    full_label
    get_options
    get_labels
    __repr__

    Examples
    --------
    Define a subclass behaving like a string, with two valid values:

    >>> class ConcreteAttribute(str, Attribute[str]):
    ...
    ...    OPTIONS = frozenset(["a", "b"])
    ...    LABELS = MappingProxyType({"a": "Alpha", "b": "Beta"})
    ...
    ...    def __new__(cls, value: str) -> Self:
    ...        if not cls.is_valid(value):
    ...            raise ValueError(f"Invalid value for {cls.__name__}: {value}")
    ...        return super().__new__(cls, value)

    Instantiate an object and get its full label:

    >>> obj = ConcreteAttribute("a")
    >>> print(obj.full_label)
    Alpha

    Check if two objects are equal:

    >>> obj1 = ConcreteAttribute("a")
    >>> obj2 = ConcreteAttribute("a")
    >>> obj3 = ConcreteAttribute("b")
    >>> print(obj1 == obj2)
    True
    >>> print(obj1 == obj3)
    False

    Check if an object belongs to a list:

    >>> items = [ConcreteAttribute("a"), ConcreteAttribute("a")]
    >>> print(obj1 in items)
    True
    >>> print(obj3 in items)
    False

    Warning
    -------
    This mixin class does not define a constructor nor a `__new__` method. It is , as expected to be
    used in a subclass which also inherits from a built-in type (e.g. `int`, `str`, `float`...).
    Each subclass should define its own constructor, calling the parent's constructor with the value
    to be stored and manipulated like the built-in type.
    """

    OPTIONS: FrozenSet[BaseT]
    LABELS: Mapping[BaseT, str]

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """
        Check if the given value an allowed option for the attribute.

        Warning
        -------
        Override in subclasses if the attribute has not a fixed set of valid options.
        """
        return value in cls.OPTIONS

    @property
    def full_label(self) -> str:
        """Full label for the value (often used for visualization)."""
        return self.LABELS.get(cast(BaseT, self), "")

    @classmethod
    def get_options(cls) -> FrozenSet[BaseT]:
        """Get all the valid options for the values of an object."""
        return cls.OPTIONS

    @classmethod
    def get_labels(cls) -> Mapping[BaseT, str]:
        """Get the full labels for the valid values."""
        return cls.LABELS

    def __getattr__(self, name: str) -> Any:
        """
        Mimic the behavior of an enumeration class to access the allowed values as attributes of the
        class (dot notation on the class name).

        Examples
        --------
        For an attribute `CoreObject` with two valid values, `a` and `b`:

        >>> a = CoreObject.a
        >>> print(a)
        a

        Parameters
        ----------
        name : str
            Name of the attribute to access.

        Returns
        -------
        Self
            Instance of the class with the value corresponding to the attribute name, if it is a
            valid option.

        Raises
        ------
        AttributeError
            If the name is not a valid option in `OPTIONS` or the parent class does not define the
            attribute (or does not implement a `__getattr__` method).

        Notes
        -----

        """
        if name in self.OPTIONS:
            return self.__class__(name)  # type: ignore[call-arg]
        if hasattr(super(), "__getattr__"):  # fallback to parent class if possible
            return super().__getattr__(name)  # type: ignore[misc]
        raise AttributeError(f"Invalid attribute for {self.__class__.__name__}: {name}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>({super().__repr__()})"

    @classmethod
    def from_container(
        cls, values: Iterable[BaseT], container: Type[Union[List, Tuple, Set]] = list
    ) -> Union[List[BaseT], Tuple[BaseT, ...], Set[BaseT]]:
        """
        Create multiple attributes from an iterable of values and store them in a container.

        Parameters
        ----------
        values : Iterable[BaseT]
            Iterable of values to create attributes from.
        container : Type[Union[List, Tuple, Set]], default=list
            Type of container to return.

        Returns
        -------
        Union[List[T], Tuple[T, ...], Set[T]]
            Container of attribute instances, of the same type as the input container.

        Raises
        ------
        ValueError
            If any of the values are invalid for this attribute type.
        """
        instances = []
        for value in values:
            if not cls.is_valid(value):
                raise ValueError(f"Invalid value for {cls.__name__}: {value}")
            # Create instance using the base type's constructor
            instance = cls.__bases__[0](value)  # assuming the first base is the desired type
            instances.append(cast(BaseT, instance))
        return container(instances)
