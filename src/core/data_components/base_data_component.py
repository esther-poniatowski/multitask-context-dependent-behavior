#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_components.data_component` [module]

Classes
-------
DataComponent
"""
from typing import Tuple, Self, FrozenSet, Dict, Type

import numpy as np
from numpy.typing import ArrayLike

from core.data_components.core_dimensions import Dimensions, DimensionsSpec


class DataComponent(np.ndarray):
    """
    Core component of a data structure, containing data values to analyze or companion labels.

    Each object behaves like a `numpy.ndarray`, with additional dimension annotations (names).

    Class Attributes
    ----------------
    DIMENSIONS_SPEC : DimensionSpec
        Names of the dimensions allowed for the data, in order, along with their status (required:
        True, optional: False).
        To be defined in subclasses.
    METADATA : FrozenSet[str]
        Names of the additional attributes storing metadata alongside with the values.
        To be defined in subclasses.
    DTYPE : np.generic
        Data type for the values stored in the array.
        To be defined in subclasses.
    SENTINEL : Any
        Sentinel value marking missing or unset values in the array.
        For float dtype: `np.nan`.
        For integer dtype: usually ``-1`` (depending on the purpose of the coordinate).
        For string dtype: usually empty string ``''``.
        To be defined in subclasses.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.dtype[np.generic]]
        Values for the underlying numpy array.
        Shape: At least the number of required dimensions, at most the total number of dimensions in
        the `DIMENSIONS_SPEC` attribute.
    dims : Dimensions, optional
        Names of each dimension in the array.
        Shape (if provided): As many elements at the number of dimensions in the input values.
        Content: Names must be consistent with the `DIMENSIONS_SPEC` attribute. The required
        dimensions must be present, and the order of dimensions should be respected.
        If not provided, default dimension names are initialized for each dimension in the input
        values.

    Methods
    -------
    __new__
    __array_finalize__
    __repr__
    default
    validate
    propagate_dimensions
    propagate_metadata
    wrap
    from_shape
    get_dim   (delegate to the `dims` attribute)
    get_axis  (delegate to the `dims` attribute)
    get_size
    get_missing
    transpose (override numpy method)
    T         (override numpy method)
    swapaxes  (override numpy method)
    moveaxis  (override numpy method)

    Examples
    --------
    Create a `DataComponent` object with default dimension names:

    >>> data = DataComponent(np.zeros((10, 5)))
    >>> data.dims
    ('', '')

    Create a `DataComponent` object with custom dimension names:

    >>> data = DataComponent(np.zeros(10, 5), dims=("units", "time"))
    >>> data.dims
    ('units', 'time')

    Get the name of the first dimension:

    >>> data.get_dim(0)
    'units'

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
    Propagation of the `dims` attribute:

    - In operations which preserve the dimensionality of the array, the `dims` attribute is
      automatically transferred from the parent array to the child array. This transfer is performed
      in the `__array_finalize__` method.
    - In operations which alter the dimensionality, the `dims` attribute is reset to the default
      value (empty strings) and must be manually updated (if necessary).

    Exceptions: Some operations preserve the dimensionality but still impact the actual
    correspondence between the data and the dimension names. Since, those operations are not
    intercepted by the checks in the `__array_finalize__` method, they are overridden.

    - In operations where the resulting dimensionality can be predicted, the `dims` attribute is
      adjusted to reflect the new structure. This is done by calling the methods provided by the
      `Dimensions` class. Operations include: `transpose`, `T`, `swapaxes`, `moveaxis`, `rollaxis`.
    - In operations where the order of elements is reversed, an error is raised to prevent the use
      of the method without updating the dimension names (`NotImplementedError`). Operations
      include: `rollaxis`, `flip` (less likely to be used in practice).
    """

    DIMENSIONS_SPEC: DimensionsSpec
    dims: Dimensions  # type hint for the instance attribute
    METADATA: FrozenSet[str]
    DTYPE: np.dtype
    SENTINEL: int | float | str

    def __repr__(self):
        if hasattr(self, "METADATA"):
            metadata = ", ".join(f"{attr}={getattr(self, attr, None)}" for attr in self.METADATA)
        else:
            metadata = ""
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"dims={getattr(self, 'dims', None)}, "
            f"{metadata})"
        )

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

        Raises
        ------
        ValueError
            If the number of dimensions in the input values does not match the number of dimensions
            in the `dim` argument.
            If the dimension names are not consistent with the `DIMENSIONS_SPEC` attribute, when
            provided.

        See Also
        --------
        `DimensionsSpec.validate`: Validate the dimension names against the `DIMENSIONS_SPEC`.
        """
        # Process input values and create the array
        cls.validate(values)  # validate input values
        # Convert input values to array - Set the data type if imposed
        if hasattr(cls, "DTYPE"):
            values = np.asarray(values, dtype=cls.DTYPE)
        else:
            values = np.asarray(values)
        obj = values.view(cls)  # cast to current class
        # Set dimensions
        if dims is None:  # default dimension names
            dims = Dimensions.default(obj.ndim)
        else:  # validate the dimension names for this class
            cls.DIMENSIONS_SPEC.validate(dims)
        if len(dims) != obj.ndim:  # check consistency between dimensions and array shape
            raise ValueError(f"len(dims) = {len(dims)} != array.ndim = {obj.ndim}")
        obj.dims = dims  # assign dimension names as new attribute
        # Set additional metadata attributes
        if hasattr(cls, "METADATA"):  # enforce specific metadata attributes
            for attr in cls.METADATA:
                setattr(obj, attr, metadata.get(attr, None))
        else:  # set any metadata attributes
            for attr, value in metadata.items():
                setattr(obj, attr, value)
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        """
        Finalize the creation of a `DataComponent` object to handle custom attributes.

        Parameters
        ----------
        obj : DataComponent
            Object to finalize.
        """
        if obj is None:  # brand-new object with no parent (in __new__)
            return
        self.propagate_dimensions(obj, self)
        self.propagate_metadata(obj, self)

    @classmethod
    def validate(cls, values: ArrayLike) -> None:
        """
        Validate input values before creating the array.

        To be overridden in subclasses for specific behavior.

        Parameters
        ----------
        values : ArrayLike
            Input values to validate.
        """
        pass

    @classmethod
    def propagate_dimensions(cls, child: np.ndarray, parent: Self) -> None:
        """
        Propagate the dimensions attribute from a parent object to a child object, if possible.

        Rules:

        - If the dimensions are preserved, transfer the attribute from the parent.
        - If the dimensions are changed, reset the attribute to the default value.

        Parameters
        ----------
        child : np.ndarray
            Child object to update with the dimensions.
        parent : DataComponent
            Parent object to propagate the dimensions from.
        """
        if parent.ndim == child.ndim:  # transfer from parent if dimensions are preserved
            dims = getattr(parent, "dims", Dimensions.default(parent.ndim))
        else:  # reset to default
            dims = Dimensions.default(child.ndim)
        setattr(child, "dims", dims)

    @classmethod
    def propagate_metadata(cls, child: np.ndarray, parent: Self) -> None:
        """
        Propagate the metadata attributes from a parent object to a child object, if possible.

        Rules:

        - By default, propagate all metadata attributes from the parent, if mentioned in the class
          attribute `METADATA`.
        - Any unpredicted instance-specific metadata attributes are not transferred.
        - Any missing metadata attributes are set to `None`.

        To be overridden in subclasses for specific behavior.

        Parameters
        ----------
        child : np.ndarray
            Child object to update with the metadata.
        parent : DataComponent
            Parent object to propagate the metadata from.
        """
        if hasattr(cls, "METADATA"):
            for attr in cls.METADATA:
                setattr(child, attr, getattr(parent, attr, None))

    def wrap(self, obj: np.ndarray) -> Self:
        """
        Cast a numpy array to a `DataComponent` object.

        Parameters
        ----------
        obj : np.ndarray
            Array to cast to the current class.

        Returns
        -------
        DataComponent
            Array cast to the current class.
        """
        if isinstance(obj, np.ndarray):
            return obj.view(type(self))
        return obj

    @classmethod
    def from_shape(
        cls, shape: int | Tuple[int, ...], dims: Dimensions | None = None, **metadata
    ) -> Self:
        """
        Create an empty instance of a `DataComponent` object with a given shape. The value used to
        mark unset or missing values is the class attribute `SENTINEL`.

        Parameters
        ----------
        shape : int | Tuple[int, ...]
            Shape of the data to create. If an integer is passed, the array will be 1D.
        dims : Dimensions, optional
            Names of the dimensions of the data.

        Returns
        -------
        DataComponent
            New instance of a `DataComponent` object, filled with the sentinel value.
        """
        if isinstance(shape, int):  # convert to tuple for consistency
            shape = (shape,)
        values = np.full(shape=shape, fill_value=cls.SENTINEL, dtype=cls.DTYPE)
        return cls(values, dims=dims, **metadata)

    # --- Getter Methods ---------------------------------------------------------------------------

    def get_dim(self, axis: int) -> str:
        """Delegate to the `dims` attribute."""
        return self.dims.get_dim(axis)

    def get_axis(self, dim: str) -> int:
        """Delegate to the `dims` attribute."""
        return self.dims.get_axis(dim)

    def get_size(self, dim: str) -> int:
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

    def get_missing(self) -> np.ndarray:
        """
        Get a boolean mask for missing values in the array, based on the sentinel value.

        Returns
        -------
        np.ndarray
            Boolean mask with `True` for missing values and `False` for valid values.
        """
        return self == self.SENTINEL

    # --- Overridden Numpy Methods -----------------------------------------------------------------

    def __getitem__(self, index) -> Self:
        """
        Get a subset of the data by indexing and cast it to a `DataComponent` object.

        Override the base numpy method to convert the result back to a `DataComponent` object.

        Parameters
        ----------
        index : Any
            Index or slice to retrieve from the data.
        """
        result = super().__getitem__(index)
        if isinstance(result, np.ndarray) and not isinstance(result, DataComponent):
            result = self.wrap(result)
            self.propagate_dimensions(result, self)
            self.propagate_metadata(result, self)
        return result

    # TODO: Decide whether to keep this method
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: np.ndarray,
        **kwargs,
    ):
        """
        Apply a universal function to the data.

        Override the base numpy method to convert the result back to a `DataComponent` object.
        Necessary for performing in-place operations without type checking errors.

        Parameters
        ----------
        ufunc : np.ufunc
            Universal function to apply.
        method : str
            Method to apply.
        inputs : np.ndarray
            Input arrays to process.
        kwargs : Any
            Additional keyword arguments for the ufunc.

        Notes
        -----
        The `__array_ufunc__` is part of the NumPy protocol for overriding ufunc behavior in custom
        array-like classes, but it is not implemented directly in numpy.ndarray.
        """
        # Convert DataComponent objects to numpy arrays
        args = [i.view(np.ndarray) if isinstance(i, DataComponent) else i for i in inputs]
        # Apply the ufunc to input arrays
        result = getattr(ufunc, method)(*args, **kwargs)
        # Convert the result back to a DataComponent object if necessary
        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, np.ndarray):
            result = self.wrap(result)
            self.propagate_dimensions(result, self)
            self.propagate_metadata(result, self)
        return result

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
        result = super().transpose(*axes)
        result.dims = self.dims.transpose(*axes)
        return result

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
        result = super().swapaxes(axis1, axis2)  # output: ndarray
        result = self.wrap(result)  # cast to the current class
        result.dims = self.dims.swap(axis1, axis2)
        return result

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
        result = np.moveaxis(self, source, destination)  # output: ndarray
        result = self.wrap(result)  # cast to the current class
        result.dims = self.dims.move(source, destination)
        return result

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


class ComponentSpec:
    """
    Specification of the data components allowed in a data structure class.

    Attributes
    ----------
    spec : Dict[str, Type[DataComponent]]
        Mapping specifying constraints for the data components.
        Keys: Attribute names for storing the data components.
        Values: Types of the data components.

    Arguments
    ---------
    kwargs : Type[DataComponent]
        Component names and types for the specification.

    Methods
    -------
    validate

    Examples
    --------
    Define a specification for the data components of a class:

    >>> spec = ComponentSpec(data=CoreData, time=CoordTime)

    Validate the data components of an object against the specification:

    >>> data = CoreData(np.zeros((10))
    >>> spec.validate("data", data)

    >>> time = CoordTime(np.zeros(10))
    >>> spec.validate("time", time)

    >>> spec.validate("data", time)
    Traceback (most recent call last):
    ...
    TypeError: Invalid component type for 'data': <class 'CoordTime'> != <class 'CoreData'>

    >>> spec.validate("units", data)
    Traceback (most recent call last):
    ...
    AttributeError: Invalid component name: 'units' not in dict_keys(['data', 'time'])

    """

    def __init__(self, **kwargs: Type[DataComponent]):
        self.spec: Dict[str, Type[DataComponent]]
        if len(set(kwargs.keys())) != len(kwargs):
            raise ValueError("Duplicate dimension names in the specification.")
        self.spec = kwargs

    def validate(self, name: str, component: DataComponent):
        """
        Validate an instance of `DataComponent` against the specification.

        Parameters
        ----------
        component : DataComponent
            Component to validate.

        Raises
        ------
        AttributeError
            If the name of the component is absent from the specification
        TypeError
            If the type of the component does not match the type expected for this name.
        """
        if name not in self.spec:
            raise AttributeError(f"Invalid component name: '{name}' not in {self.spec.keys()}.")
        if not isinstance(component, self.spec[name]):
            raise TypeError(
                f"Invalid component type for '{name}': {type(component)} != {self.spec[name]}"
            )
