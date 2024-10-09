#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.dimensions` [module]

Classes
-------
`DimName`
`Dimensions`
"""
from types import MappingProxyType
from typing import Self, Union, Iterable


class DimName(str):
    """
    Dimension names allowed in data structures.

    Class Attributes
    ----------------
    DEFAULT : str
        Default dimension name. Used for consistency in the Core Data structures when no dimension
        name is provided.
    _OPTIONS : FrozenSet
        Valid dimension names.
    _ALIASES : Mapping[str, str]
        Aliases for each dimension name. Used in properties to access their shape.

    Attributes
    ----------
    alias : str
        (Property) Alias for the dimension name.

    Notes
    -----
    Operations on this string subclass (e.g. slicing, concatenation, .upper(), .lower()...), will
    return `str` objects rather than `DimName` objects. To ensure that the result of such an
    operations is a `DimName` object, it is necessary manually convert the result back to `DimName`,
    or to override the corresponding function.
    """

    DEFAULT = ""

    _ALIASES = MappingProxyType(
        {
            DEFAULT: "",
            "spikes": "spk",
            "ensembles": "ens",
            "units": "u",
            "folds": "f",
            "trials": "tr",
            "time": "t",
        }
    )

    _OPTIONS = frozenset(_ALIASES.keys())

    def __new__(cls, value):
        if value not in cls._OPTIONS:
            raise ValueError(f"Invalid dimension name: {value}")
        return str.__new__(cls, value)

    @property
    def alias(self) -> str:
        """Get the alias for the dimension name."""
        return self._ALIASES[self]


class Dimensions(tuple):
    """
    Tuple of dimension names to store within a data structure or core data object.

    Provide utility methods to examine the dimensions, which can be used by wrapper objects via
    delegation.

    Parameters
    ----------
    args : Tuple[Union[str, DimName], ...]
        Names of the dimensions, among the valid options specified in the `DimName` class.

    Raises
    ------
    ValueError
        If any of the dimension names is invalid.

    Methods
    -------
    `get_dim`
    `get_axis`
    """

    def __new__(cls, *args: Union[str, DimName, Iterable[Union[str, DimName]]]) -> Self:
        """
        Create a new instance of the `Dimensions` class from various inputs.

        - Strings are automatically converted to `DimName`.
        - If any string is an invalid dimension name, it is detected by the `DimName` class.
        - If a single iterable is passed, it is unpacked to extract the dimension names. This allows
          to convert tuples and lists to `Dimensions` objects.
        """
        # Handle the case where a single iterable is passed
        if len(args) == 1 and isinstance(args[0], Iterable) and not isinstance(args[0], str):
            args = tuple(args[0])  # extract the single element, convert to tuple for type checking
        # Validate dimension names and via DimName
        names = tuple(DimName(arg) for arg in args)
        # Call the tuple constructor
        return super().__new__(cls, names)

    def __repr__(self) -> str:
        return f"Dimensions{super().__repr__()}"

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self)

    def get_dim(self, axis: int) -> DimName:
        """
        Get the name of a specific dimension by index.

        Parameters
        ----------
        axis : int
            Index of the axis.

        Returns
        -------
        DimName
            Name of the dimension.
        """
        if axis >= self.ndim:
            raise IndexError(f"Invalid axis: {axis} >= array.ndim {self.ndim}.")
        return self[axis]

    def get_axis(self, dim: Union[str, DimName]) -> int:
        """
        Retrieve the axis (index) corresponding to one dimension by its name.

        Parameters
        ----------
        dim : Union[str, DimName]
            Name of the dimension.

        Returns
        -------
        axis : int
            Axis number associated with the dimension.
        """
        if dim not in self:
            raise ValueError(f"Invalid dimension: '{dim}' not in {self}.")
        return self.index(dim)
