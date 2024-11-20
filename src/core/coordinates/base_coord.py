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
    ENTITY : Type[Entity]
        Entity represented by the coordinate. It determines the data type and the valid values for
        the underlying numpy array.
    DTYPE: Type[CoordDtype]
        Data type for the coordinate labels.
    METADATA : Tuple[str, ...]
        Names of the additional attributes storing metadata alongside with the coordinate values.
    SENTINEL : Any
        Sentinel value marking missing or unset coordinate values.
        For float dtype: `np.nan`.
        For integer dtype: usually ``-1`` (depending on the purpose of the coordinate).
        For string dtype: usually empty string ``''``.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any, ...], np.dtype[CoordDtype]]
        Labels of the coordinate associated with data dimension(s). Length : ``n_smpl``, total
        number of samples labelled by the coordinates across its dimensions.

    Methods
    -------
    `validate`
    `from_shape`
    `get_entity`
    `has_entity`
    `are_valid`
    """

    ENTITY: Type[EntityType]
    DTYPE: Type[CoordDtype]
    METADATA: FrozenSet[str] = frozenset()  # default empty set
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
        If necessary, override this method in subclasses for more efficient or custom tests.

        See Also
        --------
        `Entity.is_valid`
        """
        if not hasattr(cls, "ENTITY"):  # only if ENTITY is defined
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

    # --- Interaction with Entities ----------------------------------------------------------------

    @classmethod
    def get_entity(cls) -> Type[EntityType]:
        """
        Get the entity type associated with the coordinate.

        Returns
        -------
        Type[EntityType]
            Entity type associated with the coordinate.
        """
        return cls.ENTITY

    @classmethod
    def has_entity(cls, entity_type: Type[Entity]) -> bool:
        """
        Check if the coordinate is associated with one specific entity type.

        Arguments
        ---------
        entity_type : Type[Entity]
            Class bound to `Entity`, to check for.

        Returns
        -------
        bool
            True if the coordinate holds a class argument `ENTITY` of the same type or a subclass of
            `entity_type`.

        Examples
        --------
        Check for the entity `ExpFactor` for `CoordTask` (superclass of `Task` entity):

        >>> coord_task = CoordTask(["PTD", "PTD", "CLK"])
        >>> coord_task.has_entity(ExpFactor)
        True

        Check for the entity `Attention`:

        >>> coord_task.has_entity(Attention)
        False
        """
        return issubclass(cls.get_entity(), entity_type)

    @classmethod
    def are_valid(cls, values) -> np.ndarray:
        """
        Marks which values are valid for the entity type.

        Parameters
        ----------
        values : ArrayLike
            Values to check.

        Returns
        -------
        np.array
            Boolean mask indicating if the values are valid for the entity type.

        See Also
        --------
        `Entity.is_valid`
        `np.vectorize`
        """
        values = np.asarray(values)  # convert to numpy array for processing
        return np.vectorize(cls.ENTITY.is_valid)(values)  # apply element-wise
