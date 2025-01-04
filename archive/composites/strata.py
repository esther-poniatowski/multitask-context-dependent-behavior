#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.strata` [module]

Classes representing strata to group trials into subsets based on their properties.

Classes
-------
Stratum
StrataUnion
"""
from typing import Self, List, Dict, Type

from core.attributes.base_attribute import Attribute


class Stratum:
    """
    Stratum defined by the combination of several factors representing trial properties.

    Attributes
    ----------
    registry : List[str]
        Registry of the attributes names for all specified factors.
    factor_types : Dict[Type[Attribute]], str]
        Mapping from the factor types (classes) and their attribute names.

    Methods
    -------
    set_factor
    add_factor
    combine_factors
    get_factor
    __iter__
    __eq__
    __hash__
    __add__

    Notes
    -----
    Partial strata can be defined by specifying only the factors of interest. Instantiate with
    keyword arguments to avoid confusion between the factors, especially when partial strata are
    defined.

    Examples
    --------
    Initialize a partial stratum with three factors:

    >>> task = Task('PTD')
    >>> category = Category('R')
    >>> behavior = Behavior('Go')
    >>> stratum = Stratum(task=task, category=category, behavior=behavior)
    >>> stratum
    Stratum(task=PTD, category=R, behavior=Go)

    Retrieve the task:

    >>> stratum.task
    Task('PTD')

    See Also
    --------
    `Attribute`
    `StrataUnion`
    """

    # --- Create Stratum instances ------------------------------------------------------------

    def __init__(self, **factors: Attribute) -> None:
        self.registry: List[str] = []
        self.factor_types: Dict[Type[Attribute], str] = {}
        for name, factor in factors.items():
            self.set_factor(name, factor)

    def set_factor(self, name: str, factor: Attribute) -> None:
        """
        Set a new factor to the stratum after validation.

        Arguments
        ---------
        name : str
            Name of the attribute to use for the factor.
        factor : Attribute
            Factor to add to the stratum.
        """
        if not isinstance(factor, Attribute):
            raise TypeError(f"Invalid argument for Stratum: {name} not Attribute")
        setattr(self, name, factor)
        self.registry.append(name)
        self.factor_types[factor.__class__] = name

    def add_factor(self, name: str, factor: Attribute) -> Self:
        """
        Add a new factor to a stratum.

        Arguments
        ---------
        factor : Attribute
            Factor to add to the stratum.

        Returns
        -------
        new_stratum : Stratum
            New stratum instance with the added factor.

        Examples
        --------
        >>> stratum = Stratum(task=Task('PTD'), attention=Attention('a'))
        >>> new_stratum = stratum.add_factor(Category('R'))
        >>> new_stratum
        Stratum(task=PTD, attention=a, category=R)
        """
        # Copy the current stratum
        new_stratum = self.__class__(**{name: getattr(self, name) for name in self.registry})
        # Add the new factor
        new_stratum.set_factor(name, factor)
        return new_stratum

    # --- Get Stratum properties -------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Stratum({', '.join(f'{name}={fact}' for name, fact in self)})"  # use __iter__

    def get_factor(self, factor_type: Type[Attribute]) -> Attribute | None:
        """
        Get the stored value corresponding to one factor class.

        Arguments
        ---------
        factor_type : Type[Attribute]
            Class of the factor for which to retrieve the value.

        Returns
        -------
        factor : Attribute
            Value of the factor if present in the stratum instance.
            None if the factor type is not present in the stratum instance.

        Examples
        --------
        >>> stratum = Stratum(task=Task('PTD'), attention=Attention('a'))
        >>> stratum.get_factor(Task)
        Task('PTD')
        """
        name = self.factor_types.get(factor_type, None)
        if name is not None:
            return getattr(self, name, None)
        return None

    def get_attribute(self, name: str) -> Type[Attribute] | None:
        """
        Get the factor class corresponding to one factor attribute.

        Arguments
        ---------
        name : str
            Name of the  factor attribute to retrieve.

        Returns
        -------
        factor_type : Type[Attribute]
            Class of the  factor corresponding to the attribute name.
            None if the attribute name is not present in the stratum instance.

        Examples
        --------
        >>> stratum = Stratum(task=Task('PTD'), attention=Attention('a'))
        >>> stratum.get_attribute('task')
        Task
        """
        factor = getattr(self, name, None)
        factor_type = factor.__class__ if factor is not None else None
        return factor_type

    def get_attributes(self) -> List[Type[Attribute]]:
        """
        Get all the factor classes present in the stratum instance.

        Returns
        -------
        factor_types : List[Type[Attribute]]
            List of the classes of the  factors present in the stratum.

        Examples
        --------
        >>> stratum = Stratum(task=Task('PTD'), attention=Attention('a'))
        >>> stratum.get_attributes()
        [Task, Attention]
        """
        return [factor.__class__ for name, factor in self]

    # --- Magic methods ----------------------------------------------------------------------------

    def __iter__(self):
        """
        Iterate over the factors specified in the stratum.

        Yields
        ------
        name : str
            Name of the factor.
        fact : Attribute
            Factor instance.
        """
        for name in self.registry:
            yield name, getattr(self, name)

    def __eq__(self, other) -> bool:
        """
        Checks the equality between two stratum instances.

        Returns
        -------
        bool
            True if all the factors contained in both attributes are equal.

        See Also
        --------
        `Attribute.__eq__`

        Examples
        --------
        >>> stratum_1 = Stratum(task=Task('PTD'), attention=Attention('a'))
        >>> stratum_2 = Stratum(task=Task('CLK'), attention=Attention('a'))
        >>> stratum_1 == stratum_2
        False
        """
        # Check type equality
        if not isinstance(other, self.__class__):
            return False
        # Check similar names
        names_1, names_2 = set(self.registry), set(other.registry)
        if names_1 != names_2:
            return False
        # Check factors equality
        for name in names_1:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self) -> int:
        """
        Hash the stratum instance based on the values of its factors.

        Returns
        -------
        int
            Hash value of the stratum instance.
        """
        return hash(tuple(getattr(self, name) for name in self.registry))

    def __add__(self, other: Self) -> "StrataUnion":
        """
        Combine two strata into a union of strata.

        Examples
        --------
        >>> stratum_1 = Stratum(task=Task('PTD'), attention=Attention('a'))
        >>> stratum_2 = Stratum(task=Task('CLK'), attention=Attention('p'))
        >>> union = stratum_1 + stratum_2
        >>> union.get()
        [Stratum(task='PTD', attention='a'),
         Stratum(task='CLK', attention='p')]
        """
        return StrataUnion(self, other)


class StrataUnion:
    """
    Union of strata, behaving like a list of strata.

    Attributes
    ----------
    stratums :
        Arbitrary number of stratum objects.

    Methods
    -------
    to_list
    __iter__

    Raises
    ------
    TypeError
        If any of the arguments is not an `Stratum` instance.

    Examples
    --------
    >>> stratum_1 = Stratum(task=Task('PTD'), attention=Attention('a'))
    >>> stratum_2 = Stratum(task=Task('CLK'), attention=Attention('p'))
    >>> stratum_3 = Stratum(category=Category('R'), behavior=Behavior('Go'))
    >>> union = StrataUnion(stratum_1, stratum_2, stratum_3)
    >>> union.to_list()
    [Stratum(task='PTD', attention='a'),
     Stratum(task='CLK', attention='p'),
     Stratum(category='R', behavior='Go')]
    """

    def __init__(self, *stratums: Stratum) -> None:
        if any(not isinstance(cond, Stratum) for cond in stratums):
            raise TypeError(f"Invalid argument for StrataUnion: {stratums} not Stratum")
        self.stratums = list(stratums)

    def to_list(self) -> List[Stratum]:
        """Get the list of strata in the union."""
        return self.stratums

    def __iter__(self):
        """Iterate over the strata in the union, providing each stratum one by one."""
        return iter(self.stratums)
