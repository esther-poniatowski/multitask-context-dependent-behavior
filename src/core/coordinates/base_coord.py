#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.base_coord` [module]

Classes
-------
Coordinate
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=abstract-method
# Scope: `Coordinate` class
# Reason: The methods `flip` and `rollaxis` are not implemented in the base class `DataComponent`.
# --------------------------------------------------------------------------------------------------


from typing import Type, TypeVar, Generic

import numpy as np
from numpy.typing import ArrayLike

from core.data_components.base_data_component import DataComponent, Dtype
from core.attributes.base_attribute import Attribute


AnyAttribute = TypeVar("AnyAttribute", bound=Attribute)
"""Type variable for the attribute type associated with the coordinate labels."""


class Coordinate(DataComponent[Dtype], Generic[Dtype, AnyAttribute]):
    """
    Base class representing coordinates for one dimension of a data set.

    Class Attributes
    ----------------
    ATTRIBUTE : Type[Attribute]
        Attribute represented by the coordinate. It determines the data type and the valid values
        for the underlying numpy array.

    Arguments
    ---------
    values : np.ndarray[Tuple[Any, ...], np.dtype[Dtype]]
        Coordinate labels associated with data dimension(s).
    **metadata : Any
        Additional attributes to store alongside the coordinate values. The names of the attributes
        should be specified in the class attribute `METADATA`.

    Methods
    -------
    validate
    get_attribute
    has_attribute
    are_valid
    """

    ATTRIBUTE: Type[AnyAttribute]

    # --- Creation of Coordinate objects -----------------------------------------------------------

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
        if hasattr(cls, "ATTRIBUTE"):  # only if ATTRIBUTE is defined
            mask = cls.are_valid(values)
            if not np.all(mask):
                raise ValueError(f"Invalid values for {cls.__name__}")

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
