#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.base_coord` [module]

Classes
-------
Coordinate
"""

from typing import Type, TypeVar, Union, Generic, Tuple, Self, FrozenSet

import numpy as np
from numpy.typing import ArrayLike

from core.attributes.base_attribute import Attribute


Dtype = TypeVar("Dtype", bound=np.generic)
"""Type variable for the data type of the underlying numpy array."""

AnyAttribute = TypeVar("AnyAttribute", bound=Attribute)
"""Type variable for the attribute type associated with the coordinate labels."""


class Coordinate(Generic[Dtype, AnyAttribute], np.ndarray):
    """
    Base class representing coordinates for one dimension of a data set.

    Class Attributes
    ----------------
    ATTRIBUTE : Type[Attribute]
        Attribute represented by the coordinate. It determines the data type and the valid values
        for the underlying numpy array.
    DTYPE: Type[Dtype]
        Data type for the coordinate labels.
    METADATA : Tuple[str, ...]
        Names of the additional attributes storing metadata alongside with the coordinate values.
    SENTINEL : Any
        Sentinel value marking missing or unset coordinate values.
        For float dtype: `np.nan`.
        For integer dtype: usually ``-1`` (depending on the purpose of the coordinate).
        For string dtype: usually empty string ``''``.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.dtype[Dtype]]
        Labels of the coordinate associated with data dimension(s). Length : ``n_smpl``, total
        number of samples labelled by the coordinates across its dimensions.
    **metadata : Any
        Additional attributes to store alongside the coordinate values. The names of the attributes
        should be specified in the class attribute `METADATA`.

    Methods
    -------
    validate
    from_shape
    get_attribute
    has_attribute
    are_valid
    """

    ATTRIBUTE: Type[AnyAttribute]
    DTYPE: Type[Dtype]
    METADATA: FrozenSet[str] = frozenset()  # default empty set to avoid errors
    SENTINEL: Union[int, float, str]

    def __repr__(self) -> str:
        metadata_str = ", ".join(f"{attr}={getattr(self, attr, None)}" for attr in self.METADATA)
        return f"<{self.__class__.__name__}(shape={self.shape}, {metadata_str})>"

    # --- Creation of Coordinate objects -----------------------------------------------------------

    def __new__(cls, values: ArrayLike, **metadata) -> Self:
        """
        Create a new coordinate object behaving as a numpy array.

        Notes
        -----
        In NumPy subclassing, no keyword arguments can be passed to the `__new__` method should only
        directly impact the creation of the array, not being metadata attributes to store in the
        instance. Any metadata should be explicitly specified in the signature.
        """
        cls.validate(values)
        obj = np.asarray(values, dtype=cls.DTYPE).view(cls)
        for attr in cls.METADATA:
            setattr(obj, attr, metadata.get(attr, None))
        return obj

    def __array_finalize__(self, obj) -> None:
        """
        Create a new coordinate object from a previous one in a numpy operation.

        Notes
        -----
        By default, metadata is retained from the parent object. Override this method in subclasses
        to implement more fine-grained behavior.
        """
        if obj is None:
            return
        for attr in self.METADATA:
            setattr(self, attr, getattr(obj, attr, None))

    def __getitem__(self, index) -> Self:
        """
        Get a subset of the data by indexing and convert it to a `Coordinate` object.

        Parameters
        ----------
        index : Any
            Index or slice to retrieve from the data.

        Returns
        -------
        Coordinate
            Subset of the coordinate.

        Warning
        -------
        By default, all the metadata attributes are copied to the new object. Override this method
        in subclasses to implement more fine-grained behavior.
        """
        result = super().__getitem__(index)
        if isinstance(result, np.ndarray) and not isinstance(result, type(self)):
            result = result.view(type(self))  # convert back to the subclass
            for attr in self.METADATA:
                setattr(result, attr, getattr(self, attr, None))
        return result

    @classmethod
    def validate(cls, values: ArrayLike, **kwargs) -> None:
        """
        Validate the values of the coordinate.

        Default implementation: Check the values consistency with the attribute type.

        Warning
        -------
        This base implementation is appropriate for the qualitative attributes which define a
        class-level attribute `ATTRIBUTE`. For quantitative attributes, the method should be
        overridden in the subclass to implement a custom validation.

        Parameters
        ----------
        values : ArrayLike
            Values to check.
        kwargs : Any
            Any other arguments necessary for subclass-specific validation.

        Raises
        ------
        ValueError
            If any element in the values is not among the valid options for the attribute.

        Notes
        -----
        If necessary, override this method in subclasses for more efficient or custom tests.

        See Also
        --------
        `Attribute.is_valid`
        """
        if not hasattr(cls, "ATTRIBUTE"):  # only if ATTRIBUTE is defined
            mask = cls.are_valid(values)
            if not np.all(mask):
                invalid_values = np.asarray(values)[~mask]
                raise ValueError(f"Invalid values for {cls.__name__}: {invalid_values}")

    @classmethod
    def from_shape(cls, shape: int | Tuple[int, ...], **metadata) -> Self:
        """
        Create an empty coordinate object with no labels.

        Parameters
        ----------
        shape : int, Tuple[int, ...]
            Shape of the array to create. If an integer is passed, the array will be 1D.

        Returns
        -------
        Coordinate
            Instance of the subclass with an empty array as labels.
        """
        if isinstance(shape, int):  # convert to tuple for consistency
            shape = (shape,)
        values = np.full(shape=shape, fill_value=cls.SENTINEL, dtype=cls.DTYPE)
        return cls(values=values, **metadata)

    # --- Interaction with Attributes --------------------------------------------------------------

    @classmethod
    def get_attribute(cls) -> Type[AnyAttribute]:
        """
        Get the attribute type associated with the coordinate.

        Returns
        -------
        Type[AnyAttribute]
            Attribute type associated with the coordinate.
        """
        return cls.ATTRIBUTE

    @classmethod
    def has_attribute(cls, attribute_type: Type[Attribute]) -> bool:
        """
        Check if the coordinate is associated with one specific attribute type.

        Arguments
        ---------
        attribute_type : Type[Attribute]
            Class bound to `Attribute`, to check for.

        Returns
        -------
        bool
            True if the coordinate holds a class argument `ATTRIBUTE` of the same type or a subclass
            of `attribute_type`.

        Examples
        --------
        Check for the attribute `ExpFactor` for `CoordTask` (superclass of `Task` attribute):

        >>> coord_task = CoordTask(["PTD", "PTD", "CLK"])
        >>> coord_task.has_attribute(ExpFactor)
        True

        Check for the attribute `Attention`:

        >>> coord_task.has_attribute(Attention)
        False
        """
        return issubclass(cls.get_attribute(), attribute_type)

    @classmethod
    def are_valid(cls, values) -> np.ndarray:
        """
        Marks which values are valid for the attribute type.

        Parameters
        ----------
        values : ArrayLike
            Values to check.

        Returns
        -------
        np.array
            Boolean mask indicating if the values are valid for the attribute type.

        See Also
        --------
        `Attribute.is_valid`
        `np.vectorize`
        """
        values = np.asarray(values)  # convert to numpy array for processing
        return np.vectorize(cls.ATTRIBUTE.is_valid)(values)  # apply element-wise
