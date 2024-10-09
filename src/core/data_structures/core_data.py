#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.core_data` [module]

Classes
-------
`DimName`
`CoreData`
"""
from types import MappingProxyType
from typing import Tuple, Optional, Self, Union, Iterable

import numpy as np


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
        Names of the dimensions. Each name can be either a valid string matching a dimension name,
        or a `DimName` object.

    Raises
    ------
    ValueError
        If any of the dimension names is invalid.

    Methods
    -------
    `get_dim`
    `get_axis`

    Notes
    -----
    - Strings are automatically converted to `DimName`.
    - If any string is an invalid dimension name, it is detected by the `DimName` class.
    """

    def __new__(cls, *args):
        names = tuple(DimName(arg) for arg in args)
        return super().__new__(cls, names)

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


class CoreData(np.ndarray):
    """
    Core component of a data structure, containing the actual data values to analyze.

    Each object behaves like a `numpy.ndarray`, with additional dimension annotations (names).

    Attributes
    ----------
    dims : Tuple[DimName, ...]
        Names of each dimension in the numpy array.

    Methods
    -------
    __new__
    __array_finalize__
    __repr__
    default_dims
    get_dim   (delegate to the `dims` attribute)
    get_axis  (delegate to the `dims` attribute)
    get_size
    transpose (override numpy method)
    T         (override numpy method)
    swapaxes  (override numpy method)
    moveaxis  (override numpy method)

    Examples
    --------
    Create a `CoreData` object with default dimension names:

    >>> data = CoreData(np.zeros((10, 5)))
    >>> data.dims
    (DimName(''), DimName(''))

    Create a `CoreData` object with custom dimension names:

    >>> data = CoreData(np.zeros(10, 5), dims=("units", "time"))
    >>> data.dims
    (DimName('units'), DimName('time'))

    Get the name of the first dimension:

    >>> data.get_dim(0)
    DimName('units')

    Get the index of the dimension named "time":

    >>> data.get_axis("time")
    1

    Get the length of the dimension named "units":

    >>> data.get_size("units")
    10

    See Also
    --------
    :class:`np.ndarray`: Numpy array class from which this class inherits.

    Notes
    -----
    The `__array_finalize__` method is called automatically after any operation that creates a new
    array (or view) from the current one. It serves to handle the custom attribute `dims` in the new
    object based on the parent object.

    In operations that alter the shape or dimensionality of the array, the attribute `dims` should
    not be preserved from the parent since they does not reflect the structure of the new object.
    The attribute is transferred from the parent array only if the dimensions are preserved.
    Otherwise, it is reset to the default value (empty strings) and must be manually updated.

    Exceptions (not intercepted by the check in `__array_finalize__`):

    - Operations which swap the axes but preserve the dimensions: `transpose`, `T`, `swapaxes`,
      `moveaxis`, `rollaxis`.
    - Operations which reverse the order of elements and may impact the actual correspondence
      between the data and the dimension names: `flip`.

    Those methods are either overridden to update the dimension names accordingly, or marked as not
    implemented to prevent their usage (for methods which are less likely to be used in practice).
    """

    # Declare custom attributes at the class level to specify type hints
    dims: Dimensions

    def __new__(
        cls,
        values: Union[Iterable, np.ndarray],
        dims: Optional[Union[Dimensions, Tuple[Union[str, DimName], ...]]] = None,
    ) -> Self:
        """
        Create a new instance of a `CoreData` object.

        Parameters
        ----------
        values : Union[Iterable, np.ndarray]
            Values to store in the object.
        dims : Tuple[str, ...]
            Names of the dimensions of the data.

        Returns
        -------
        CoreData
            New instance of a `CoreData` object.
        """
        obj = np.asarray(values).view(cls)  # create NumPy array and cast to current class
        if dims is None:
            dims = cls.default_dims(obj)
        if len(dims) != obj.ndim:
            raise ValueError(f"len(dims) = {len(dims)} != array.ndim = {obj.ndim}")
        obj.dims = Dimensions(*dims)  # unpack tuple, automatic validation by `Dimensions`
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        """
        Finalize the creation of a `CoreData` object to handle the custom attribute `dims`.

        Rules:

        - If the dimension is preserved, transfer the attribute from the initial object.
        - If the dimension is changed, reset the attribute to the default value.

        Parameters
        ----------
        obj : CoreData
            Object to finalize.
        """
        if obj is None:  # brand-new object with no parent (__new__)
            return
        if obj.ndim == self.ndim:  # transfer from parent
            self.dims = getattr(obj, "dims", self.default_dims(obj))
        else:  # reset to default
            self.dims = self.default_dims(self)

    def __repr__(self):
        return f"CoreData(shape={self.shape}, dims={self.dims}, array={super().__repr__()})"

    @staticmethod
    def default_dims(obj) -> Dimensions:
        """
        Create default dimension names: one empty string per dimension in the object.

        Parameters
        ----------
        obj : CoreData
            Object to create dimension names for.

        Returns
        -------
        Tuple[DimName, ...]
            Default dimension names.
        """
        return Dimensions(DimName.DEFAULT) * obj.ndim

    def __getattr__(self, name):
        """Delegate the `get_dim` and `get_axis` methods to the `dims` attribute."""
        if name == "get_dim":
            return self.dims.get_dim
        if name == "get_axis":
            return self.dims.get_axis

    def get_size(self, dim: Union[str, DimName]) -> int:
        """
        Retrieve the length of a dimension.

        Parameters
        ----------
        dim : Union[str, DimName]s
            Name of the dimension.

        Returns
        -------
        n : int
            Number of elements along the dimension.
        """
        return self.shape[self.get_axis(dim)]  # automatic check of the dimension

    def transpose(self, *axes) -> Self:
        """
        Transpose and swap dimension names to match the new axis order.

        Arguments
        ---------
        axes : Tuple[int, ...]
            New order of the axes. If empty, the axes are reversed. Otherwise, the integer at each
            position indicates the new position of the axis whose current position is the index.

        See Also
        --------
        :meth:`np.ndarray.transpose`: Numpy method to transpose arrays.
        """
        transposed_array = super().transpose(*axes)
        if len(axes) == 0:  # default transpose: reverses axes
            transposed_array.dims = Dimensions(*self.dims[::-1])  # unpack tuple
        else:  # custom axis order: swap dimension names
            transposed_array.dims = Dimensions(self.dims[i] for i in axes)
        return transposed_array

    @property
    def T(self):  # pylint: disable=invalid-name
        """Redirect to the custom `transpose` method."""
        return self.transpose()

    def swapaxes(self, axis1, axis2) -> Self:
        """
        Swap two axes and adjust dimension names accordingly.

        Arguments
        ---------
        axis1, axis2 : int
            Indices of the axes to swap.

        See Also
        --------
        :meth:`np.ndarray.swapaxes`: Numpy method to swap two axes.
        """
        swapped_array = super().swapaxes(axis1, axis2)
        swapped_dims = list(self.dims)  # convert tuple to list to modify
        swapped_dims[axis1], swapped_dims[axis2] = swapped_dims[axis2], swapped_dims[axis1]
        swapped_array.dims = tuple(swapped_dims)  # type: ignore[attr-defined]
        return swapped_array  # type: ignore[return-value]

    def moveaxis(self, source, destination) -> Self:
        """
        Move specified axes to new positions and adjust dimension names accordingly.

        Arguments
        ---------
        source : Union[int, Tuple[int, ...]]
            Indices of the axes to move.
        destination : Union[int, Tuple[int, ...]]
            New positions of the axes.

        See Also
        --------
        :meth:`np.moveaxis`: Numpy method to move axes to new positions.
        """
        moved_array = np.moveaxis(self, source, destination)
        moved_dims = list(self.dims)  # convert tuple to list to modify
        if isinstance(source, int):
            source = [source]
        if isinstance(destination, int):
            destination = [destination]
        for src, dest in zip(source, destination):
            moved_dims.insert(dest, moved_dims.pop(src))  # remove src and insert at dest
        moved_array.dims = tuple(moved_dims)  # type: ignore[attr-defined]
        return moved_array  # type: ignore[return-value]

    def rollaxis(self, axis, start=0) -> Self:
        """
        Roll an axis backwards to the given position and adjust dimension names accordingly.

        See Also
        --------
        :meth:`np.rollaxis`: Numpy method to roll an axis to a new position.
        """
        raise NotImplementedError("Update the dimension names accordingly.")

    def flip(self, axis) -> Self:
        """
        Reverse the order of elements along an axis and adjust dimension names accordingly.

        See Also
        --------
        :meth:`np.flip`: Numpy method to reverse the order of elements along an axis.
        """
        raise NotImplementedError("Update the dimension names accordingly.")
