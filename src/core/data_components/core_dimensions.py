#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_components.core_dimensions` [module]

Classes
-------
Dimensions
DimensionsSpec
"""
from collections import UserList, OrderedDict
from typing import Self, Tuple


class Dimensions(UserList):
    """
    Dimension names to label the axes of a data component or data structure.

    Provide utility methods to examine the dimensions, which can be used by wrapper objects via
    delegation.

    Class Attributes
    ----------------
    DEFAULT : str
        Default dimension name. Used for consistency in CoreData objects when no dimension name is
        provided.

    Attributes
    ----------
    args : str
        Names of the dimensions.

    Methods
    -------
    default
    get_dim
    get_axis
    is_subset
    is_ordered_as
    add
    transpose

    Examples
    --------
    Create a new dimension name:

    >>> dims = Dimensions("time", "trials", "units")

    Get the number of dimensions:

    >>> dims.ndim
    3

    Get the name of a specific dimension by index:

    >>> dims.get_dim(0)
    "time"

    Retrieve the axis (index) corresponding to one dimension by its name:

    >>> dims.get_axis("trials")
    1

    Check if the dimensions are a subset of another set of dimensions:

    >>> partial_dims = Dimensions("time", "trials")
    >>> partial_dims.is_subset(full_dims))
    True

    See Also
    --------
    `collections.UserList` : Inherit from this class to provide list-like behavior to the object.
    """

    DEFAULT = ""

    # --- Creation of Dimensions -------------------------------------------------------------------

    def __init__(self, *args: str) -> None:
        # Check uniqueness of dimension names
        if len(set(args)) != len(args):
            raise ValueError(f"Duplicate dimension names: {args}")
        # Call parent constructor (UserList)
        super().__init__(args)

    def __repr__(self) -> str:
        return f"Dimensions{super().__repr__()}"

    @classmethod
    def default(cls, ndim: int) -> Self:
        """
        Create a default set of dimensions with a given number of dimensions.

        Parameters
        ----------
        ndim : int
            Number of dimensions.

        Returns
        -------
        Dimensions
            Default set of dimensions.
        """
        return cls(*[cls.DEFAULT for _ in range(ndim)])

    # --- Getter methods ---------------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self)

    def get_dim(self, axis: int) -> str:
        """
        Get the name of a specific dimension by index.

        Parameters
        ----------
        axis : int
            Index of the axis.

        Returns
        -------
        str
            Name of the dimension.

        Raises
        ------
        IndexError
            If the axis is out of bounds.
        """
        if axis >= self.ndim:
            raise IndexError(f"Invalid dimension index: {axis} >= ndim {self.ndim}.")
        return self[axis]

    def get_axis(self, name: str) -> int:
        """
        Retrieve the axis (index) corresponding to one dimension by its name.

        Parameters
        ----------
        name : str
            Name of the dimension.

        Returns
        -------
        axis : int
            Axis number associated with the dimension.

        Raises
        ------
        ValueError
            If the dimension name is not among the dimensions.
        """
        if not name in self:
            raise ValueError(f"Invalid dimension name: '{name}' not among the dimensions.")
        return self.index(name)

    # --- Comparison methods -----------------------------------------------------------------------

    def is_subset(self, other: Self) -> bool:
        """
        Check if the dimensions are a subset of another set of dimensions.

        Parameters
        ----------
        other : Dimensions
            Other set of dimensions.

        Returns
        -------
        bool
            True if the dimensions are a subset of the other set of dimensions.
        """
        return set(self).issubset(set(other))

    def is_ordered_as(self, other: Self) -> bool:
        """
        Check if the dimensions are in the same order as another set of dimensions.

        The check only considers the dimensions which are common to both objects.

        Parameters
        ----------
        other : Dimensions
            Other set of dimensions to compare with.

        Returns
        -------
        bool
            True if the common dimensions are in the same order as the other set of dimensions.
        """
        common_dims = set(self) & set(other)
        order_self = list(self[i] for i in range(len(self)) if self[i] in common_dims)
        order_other = list(other[i] for i in range(len(other)) if other[i] in common_dims)
        return order_self == order_other

    # --- Manipulation methods ---------------------------------------------------------------------

    def add(self, name: str = DEFAULT, axis: int = -1) -> None:
        """
        Add a dimension, to a specific position if needed.

        Arguments
        ---------
        name : str, optional
            Name of the dimension, by default DEFAULT.
        axis : int, optional
            Index of the axis, by default -1 (last position).

        Raises
        ------
        ValueError
            If the dimension name already exists.
        IndexError
            If the axis is out of bounds.
        """
        if name in self:
            raise ValueError(f"Duplicate dimension name: '{name}' already exists.")
        if axis < 0:
            axis = self.ndim + axis + 1
        if axis < 0 or axis > self.ndim:
            raise IndexError(f"Invalid axis index: {axis} out of bounds for existing dimensions.")
        self.insert(axis, name)

    def transpose(self, axes: Tuple[int, ...] | None = None) -> Self:
        """
        Reverse the order of the dimensions.

        Arguments
        ---------
        axes : Tuple[int, ...]
            Indices of the dimensions in the new order.
            If empty, the axes are reversed.
            Otherwise, the integer at each position indicates the new position of the axis currently
            at this index.

        Returns
        -------
        Dimensions
            New instance with the dimensions in reverse order.

        Examples
        --------
        Reverse the order of the dimensions:

        >>> dims = Dimensions("units", "trials", "time")
        >>> dims.transpose()
        Dimensions("time", "trials", "units")

        Change the order of the dimensions:

        >>> dims.transpose((2, 0, 1))
        Dimensions("time", "units", "trials")

        Explanation:

        - The first dimension (index 0) is moved to the last position (index 2).
        - The second dimension (index 1) is kept in the middle.
        - The last dimension (index 2) is moved to the first position (index 0).
        """
        if axes is None:  # reverse order
            axes = tuple(range(self.ndim - 1, -1, -1))  # range(start, stop, step), start > stop
        return self.__class__(*[self[axis] for axis in axes])

    def swap(self, axis1: int, axis2: int) -> Self:
        """
        Swap the position of two dimensions.

        Parameters
        ----------
        axis1 : int
            Index of the first dimension.
        axis2 : int
            Index of the second dimension.

        Returns
        -------
        Dimensions
            New instance with the two dimensions swapped.

        Raises
        ------
        IndexError
            If any of the indices is out of bounds.
        """
        for axis in [axis1, axis2]:
            if axis < 0 or axis >= self.ndim:
                raise IndexError(f"Invalid axis index for {self.ndim} dimensions: {axis}.")
        new_dims = list(self)  # convert to list to modify
        new_dims[axis1], new_dims[axis2] = new_dims[axis2], new_dims[axis1]
        return self.__class__(*new_dims)

    def move(self, source, destination):
        """
        Move specified dimensions to new positions.

        Arguments
        ---------
        source : Union[int, Tuple[int, ...]]
            Indices of the axes to move.
        destination : Union[int, Tuple[int, ...]]
            New positions of the axes.

        Returns
        -------
        Dimensions
            New instance with the dimensions moved.
        """
        # Convert single index inputs to list for consistency
        if isinstance(source, int):
            source = [source]
        if isinstance(destination, int):
            destination = [destination]
        new_dims = list(self)  # convert to list to modify
        for src, dest in zip(source, destination):
            new_dims.insert(dest, new_dims.pop(src))  # remove source, insert at destination
        return self.__class__(*new_dims)


class DimensionsSpec:
    """
    Specification for dimension names in a pipeline or nested object.

    Attributes
    ----------
    spec : OrderedDict[str, bool]
        Ordered mapping specifying constraints for the dimensions.
        Keys (str): Dimension names allowed in the instances of the `Dimensions` class.
        Values (bool): Boolean indicating if the dimension is required (True) or optional (False).
        The order of the keys determines the expected order of the dimensions in the instances.

    Methods
    -------
    validate
    required
    optional
    order
    """

    def __init__(self, spec: OrderedDict[str, bool]) -> None:
        if len(set(spec.keys())) != len(spec):
            raise ValueError("Duplicate dimension names in the specification.")
        self.spec = spec

    def required(self) -> Dimensions:
        """Get the required dimensions in an instance of the `Dimensions` class."""
        return Dimensions(*[dim for dim, required in self.spec.items() if required])

    def optional(self) -> Dimensions:
        """Get the optional dimensions in an instance of the `Dimensions` class."""
        return Dimensions(*[dim for dim, required in self.spec.items() if not required])

    def validate(self, dims: Dimensions) -> None:
        """
        Validate an instance of Dimensions against the specification.

        Parameters
        ----------
        dims : Dimensions
            Dimensions to validate.

        Raises
        ------
        ValueError
            If any required dimensions is missing.
            If any extra dimension is present.
            If the order of the dimensions is incorrect.
        """
        dims_set = set(dims)
        spec_set = set(self.spec.keys())
        # Check dimension names
        missing = [dim for dim, required in self.spec.items() if required and dim not in dims_set]
        extra = [dim for dim in dims_set if dim not in spec_set]
        if missing:
            raise ValueError(f"Missing required dimensions: {missing}")
        if extra:
            raise ValueError(f"Extra dimensions not allowed by the specification: {extra}")
        # Check order
        spec_order = [dim for dim in self.spec.keys() if dim in dims]  # common dimensions
        actual_order = list(dims)
        if spec_order != actual_order:
            raise ValueError(f"Incorrect order: {actual_order} instead of {spec_order}.")
