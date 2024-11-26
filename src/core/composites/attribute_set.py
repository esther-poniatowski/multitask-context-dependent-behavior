#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.attribute_set` [module]

Classes representing sets of attributes which jointly specify a category or condition.

Classes
-------
AttributeSet
AttributeSetUnion
"""
from typing import Self, List, Type

from core.attributes.base_attribute import Attribute
from core.composites.base_container import Container


class AttributeSet(Container[Type[Attribute], Attribute]):
    """
    Combination of attributes which jointly specifies a category or condition.

    Class Attributes
    ----------------
    KEY_TYPE : Type[Attribute]
        Type of the keys in the container, here types of attributes (classes).
    VALUE_TYPE : Attribute
        Type of the values in the container, here attributes instances.

    Attributes
    ----------
    See the base class `Container`.

    Arguments
    ---------
    *args : Attribute
        Attribute values to include in the set (instances of an  `Attribute` subclass). The key of
        each item (i.e. its type) is automatically inferred from the class of the attribute.

    Methods
    -------
    set
    add
    get (inherited from `UserDict`, see in "See Also")
    __iter__
    __add__

    Warning
    -------
    This container can only store a single value per attribute type. If an attribute of the
    a pre-existing type is added in the set, this new attribute will replace the old one.
    This behavior is relevant since each trial or condition is described by a unique value for each
    attribute.

    Examples
    --------
    Initialize a set with three experimental factors:

    >>> set = AttributeSet(Task('PTD'), Category('R'), Behavior('Go'))
    >>> set
    AttributeSet(Behavior('Go'), Category('R'), Task('PTD'))

    Retrieve the value of the `Task` attribute stored in the set:

    >>> set.get(Task)
    Task('PTD')

    See Also
    --------
    `Attribute`
    `AttributeSetUnion`
    `collections.UserDict.get`
        Retrieve the value associated with a specified key from the dictionary.
        Usage: ``value = userdict.get(key, default=None)``
    """

    KEY_TYPE = type(Attribute)
    VALUE_TYPE = Attribute

    def __repr__(self) -> str:
        attr_classes = sorted(self.keys(), key=lambda cls: cls.__name__)  # order by names
        return f"{self.__class__.__name__}({', '.join(f'{self[cls]}' for cls in attr_classes)})"

    def __init__(self, *args: Attribute) -> None:
        """Override the base constructor to fix `key_type` and `value_type`."""
        # Create the dictionary of attributes with the class of the attribute as key
        data = {feature.__class__: feature for feature in args}
        # Pass data as positional argument to the parent class with fixed key and value types
        super().__init__(data, key_type=self.KEY_TYPE, value_type=self.VALUE_TYPE)

    def set(self, value: Attribute) -> None:
        """
        Set a new attribute in the set, with a key corresponding to the class of the attribute.

        Arguments
        ---------
        value : Attribute
            Attribute instance to set in the set.
        """
        key = value.__class__
        self[key] = value

    def add(self, value: Attribute) -> Self:
        """
        Create a new instance of the set with an additional attribute.

        Arguments
        ---------
        value : Attribute

        Returns
        -------
        AttributeSet
            New instance of the set with the additional attribute.
        """
        new_set = self.copy()
        new_set.set(value)
        return new_set

    def __add__(self, other: Self) -> "AttributeSetUnion":
        """
        Combine two sets of attributes into a union of sets of attributes.

        Override the default behavior of the `+` operator to combine two dictionaries, which would
        instead merge the dictionaries.

        Examples
        --------
        >>> stratum_1 = AttributeSet(Task('PTD'), Attention('a'))
        >>> stratum_2 = AttributeSet(Task('CLK'), Attention('p'))
        >>> union = stratum_1 + stratum_2
        >>> union.to_list()
        [AttributeSet(Task('PTD'), Attention('a')), AttributeSet(Task('CLK'), Attention('p'))]
        """
        return AttributeSetUnion(self, other)


class AttributeSetUnion:
    """
    Union of sets of attributes, behaving like a list of sets.

    Attributes
    ----------
    sets : List[AttributeSet]
        Arbitrary number of attribute sets.

    Methods
    -------
    to_list
    __iter__

    Raises
    ------
    TypeError
        If any of the arguments is not an `AttributeSet` instance.

    Examples
    --------
    >>> stratum_1 = AttributeSet(Task('PTD'), Attention('a'))
    >>> stratum_2 = AttributeSet(Task('CLK'), Attention('p'))
    >>> stratum_3 = AttributeSet(Category('R'), Behavior('Go'))
    >>> union = AttributeSetUnion(stratum_1, stratum_2, stratum_3)
    >>> union.to_list()
    [AttributeSet(Task('PTD'), Attention('a')),
     AttributeSet(Task('CLK'), Attention('p')),
     AttributeSet(Category('R'), Behavior('Go'))]
    """

    def __init__(self, *attribute_sets: AttributeSet) -> None:
        for attr_set in attribute_sets:
            if not isinstance(attr_set, AttributeSet):
                raise TypeError(
                    f"Invalid argument for AttributeSetUnion: {attr_set} not AttributeSet"
                )
        self.sets = list(attribute_sets)

    def to_list(self) -> List[AttributeSet]:
        """Get the list of sets of attributes in the union."""
        return self.sets

    def __iter__(self):
        """Iterate over the sets of attributes in the union, providing each set one by one."""
        return iter(self.sets)
