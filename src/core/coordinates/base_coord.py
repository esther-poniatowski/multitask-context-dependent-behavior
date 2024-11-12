#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.base_coord` [module]

Classes
-------
`Coordinate`
"""

from typing import Type, TypeVar, Union, Generic, Tuple, Self, FrozenSet

import numpy as np
from numpy.typing import ArrayLike

from core.entities.base_entity import Entity


CoordDtype = TypeVar("CoordDtype", bound=np.generic)
"""Type variable for the labels of the coordinate."""

EntityType = TypeVar("EntityType", bound=Entity)
"""Type variable for the entity type associated with the coordinate."""


class Coordinate(Generic[CoordDtype, EntityType], np.ndarray):
    """
    Base class representing coordinates for one dimension of a data set.

    Class Attributes
    ----------------
    entity : Type[Entity]
        Entity represented by the coordinate. It determines the data type and the valid values for
        the underlying numpy array.
    dtype: Type[CoordDtype]
        Data type for the coordinate labels.
    metadata : Tuple[str, ...]
        Names of the additional attributes storing metadata alongside with the coordinate values.
    sentinel : Any
        Sentinel value marking missing or unset coordinate values.
        For float dtype: `np.nan`.
        For integer dtype: usually ``-1`` (depending on the purpose of the coordinate).
        For string dtype: usually empty string ``''``.

    Attributes
    ----------
    values : npt.ndarray[Tuple[Any, ...], np.dtype[CoordDtype]]
        Labels of the coordinate associated with data dimension(s). Length : ``n_smpl``, total
        number of samples labelled by the coordinates across its dimensions.

    Methods
    -------
    `validate`
    `from_shape`
    """

    ENTITY: Type[EntityType]
    DTYPE: Type[CoordDtype]
    METADATA: FrozenSet[str] = frozenset()  # default empty set
    SENTINEL: Union[int, float, str]

    def __new__(cls, values: ArrayLike) -> Self:
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
        return obj

    def set_metadata(self, **metadata) -> None:
        """
        Set metadata attributes from a dictionary.

        Notes
        -----
        Metadata is added if it corresponds to the attribute names specified in the class-level
        attribute.
        Otherwise, the corresponding instance attributes are initialized to `None`.
        """
        for attr in self.METADATA:
            setattr(self, attr, metadata.get(attr, None))

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

    @classmethod
    def validate(cls, values: ArrayLike, **kwargs) -> None:
        """
        Check the values consistency with the entity type.

        Parameters
        ----------
        values : ArrayLike
            Values to check.
        kwargs : Any
            Any other arguments necessary for subclass-specific validation.

        Raises
        ------
        ValueError
            If any element in the values is not among the valid options for the entity.

        Notes
        -----
        Override this method in subclasses for more efficient tests if necessary.
        """
        values = np.asarray(values)  # convert to numpy array for processing
        is_valid = np.vectorize(cls.ENTITY.is_valid)(values)  # apply element-wise
        if not np.all(is_valid):
            raise ValueError(f"Invalid values for {cls.__name__}: {values[~is_valid]}")

    @classmethod
    def from_shape(cls, shape: Tuple[int, ...], **metadata) -> Self:
        """
        Create an empty coordinate object with no labels.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the array to create.

        Returns
        -------
        Coordinate
            Instance of the subclass with an empty array as labels.
        """
        values = np.full(shape, cls.SENTINEL, dtype=cls.DTYPE)
        return cls(values=values, **metadata)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> : {len(self)} samples."
