#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_components.data_component` [module]

Classes
-------
DataComponent
"""
from typing import Tuple, Self, Iterable, TypeVar, Type

import numpy as np
from numpy.typing import ArrayLike

from core.data_components.core_dimensions import Dimensions

Dtype = TypeVar("Dtype", bound=np.generic)
"""Type variable for the data type of the underlying numpy array."""


class DataComponent(np.ndarray):
    """
    Core component of a data structure, containing data values to analyze or companion labels.

    Each object behaves like a `numpy.ndarray`, with additional dimension annotations (names).

    Class Attributes
    ----------------
    DIMS : OrderedDict[str, bool]
        Names of the dimensions allowed for the data, in order. The boolean value indicates whether
        this dimension is required (True) or optional (False).
        To be defined in subclasses.
    METADATA : Tuple[str, ...]
        Names of the additional attributes storing metadata alongside with the values. To be defined
        in subclasses.
    DTYPE : np.generic
        Data type for the values stored in the array. To be defined in subclasses.
    SENTINEL : Any
        Sentinel value marking missing or unset values in the array.
        For float dtype: `np.nan`.
        For integer dtype: usually ``-1`` (depending on the purpose of the coordinate).
        For string dtype: usually empty string ``''``.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.dtype[Dtype]]
        Values for the underlying numpy array.
        Shape:
    dims : Dimensions
        Names of each dimension in the array.

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
    Create a `DataComponent` object with default dimension names:

    >>> data = DataComponent(np.zeros((10, 5)))
    >>> data.dims
    (DimName(''), DimName(''))

    Create a `DataComponent` object with custom dimension names:

    >>> data = DataComponent(np.zeros(10, 5), dims=("units", "time"))
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
    `numpy.ndarray`: Numpy array class from which this class inherits.

    Notes
    -----
    The

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

    DIMS: Dimensions
    METADATA: Tuple[str, ...]
    DTYPE: Type[Dtype]
    SENTINEL: int | float | str

    # --- Initialization Methods -------------------------------------------------------------------

    def __new__(cls, values: ArrayLike, dims: Dimensions | None = None, **metadata) -> Self:
        """
        Create a new instance of a `DataComponent` object behaving as a numpy array.

        Parameters
        ----------
        values : ArrayLike
            Values to store in the object.
        dims : Dimensions | Tuple[str | DimName, ...], optional
            Names of the dimensions of the data.

        Returns
        -------
        DataComponent
            New instance of a `DataComponent` object.
        """
        # Process input values and create the array
        cls.validate(values)  # check the validity of the input values
        if hasattr(cls, "DTYPE"):  # set the dtype if the class imposes
            values = np.asarray(values, dtype=cls.DTYPE)
        else:
            values = np.asarray(values)
        obj = values.view(cls)  # create NumPy array and cast to current class
        # Set dimension attributes
        if dims is None:  # default dimension names
            dims = cls.default_dims(obj)
        if len(dims) != obj.ndim:  # check consistency between dimensions and array shape
            raise ValueError(f"len(dims) = {len(dims)} != array.ndim = {obj.ndim}")
        obj.dims = dims  # assign dimension names as new attribute
        # Set additional metadata attributes
        if hasattr(cls, "METADATA"):  # enforce specific metadata attributes if defined
            for attr in cls.METADATA:
                setattr(obj, attr, metadata.get(attr, None))
        else:  # set any additional metadata attributes
            for attr, value in metadata.items():
                setattr(obj, attr, value)
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        """
        Finalize the creation of a `DataComponent` object to handle the custom attribute `dims`.

        Rules:

        - If the dimension is preserved, transfer the attribute from the initial object.
        - If the dimension is changed, reset the attribute to the default value.

        Parameters
        ----------
        obj : DataComponent
            Object to finalize.
        """
        if obj is None:  # brand-new object with no parent (in __new__)
            return
        if obj.ndim == self.ndim:  # transfer from parent if dimensions are preserved
            self.dims = getattr(obj, "dims", self.default_dims(obj))
        else:  # reset to default
            self.dims = self.default_dims(self)

    def __getitem__(self, index) -> Self:
        """
        Get a subset of the data by indexing and convert it to a `DataComponent` object.

        Overridden to convert the result back to a `DataComponent` object if it is a NumPy array.

        Parameters
        ----------
        index : Any
            Index or slice to retrieve from the data.
        """
        result = super().__getitem__(index)
        if isinstance(result, np.ndarray) and not isinstance(result, DataComponent):
            result = result.view(type(self))  # convert back to DataComponent
            if result.ndim == self.ndim:
                result.dims = self.dims
            else:
                result.dims = self.default_dims(result)
        return result

    @classmethod
    def from_shape(
        cls, shape: Tuple[int, ...], dims: Dimensions | Tuple[str | DimName, ...] | None = None
    ) -> Self:
        """
        Create an empty instance of a `DataComponent` object with a given shape.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the data to create.
        dims : Dimensions | Tuple[str | DimName, ...], optional
            Names of the dimensions of the data.

        Returns
        -------
        DataComponent
            New instance of a `DataComponent` object. Empty values are filled with NaNs.
        """
        return cls(np.full(shape, np.nan), dims)

    def __repr__(self):
        return f"DataComponent(shape={self.shape}, dims={self.dims}, array={super().__repr__()})"

    @staticmethod
    def default_dims(obj) -> Dimensions:
        """
        Create default dimension names: one empty string per dimension in the object.

        Parameters
        ----------
        obj : DataComponent
            Object to create dimension names for.

        Returns
        -------
        Dimensions
            Default dimension names.
        """
        return Dimensions(DimName.DEFAULT) * obj.ndim

    # --- Utility Methods --------------------------------------------------------------------------

    def get_dim(self, axis: int) -> DimName:
        """Delegate to the `dims` attribute."""
        return self.dims.get_dim(axis)

    def get_axis(self, dim: str | DimName) -> int:
        """Delegate to the `dims` attribute."""
        return self.dims.get_axis(dim)

    def get_size(self, dim: str | DimName) -> int:
        """
        Retrieve the length of a dimension.

        Parameters
        ----------
        dim : str | DimName
            Name of the dimension.

        Returns
        -------
        n : int
            Number of elements along the dimension.
        """
        return self.shape[self.get_axis(dim)]  # automatic check of the dimension

    # --- Overridden Numpy Methods -----------------------------------------------------------------

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
            transposed_array.dims = Dimensions(self.dims[::-1])
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
