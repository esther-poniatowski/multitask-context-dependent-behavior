#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.coordinates.base` [module]

Classes
-------
:class:`Coordinate`
"""

from typing import Any, Type, TypeVar, Union, Generic, overload, Self

from abc import ABC, abstractmethod
import copy
import numpy as np
import numpy.typing as npt

from utils.misc.functions import filter_kwargs


C = TypeVar("C", bound="Coordinate")
"""Type variable for Coordinate subclasses."""

L = TypeVar("L", np.int64, np.float64, np.str_)
"""Type variable for the labels of the coordinate."""


class Coordinate(Generic[L], ABC):
    """
    Abstract Base Class representing coordinates for one dimension of a data set.

    Attributes
    ----------
    values: npt.NDArray[L]
        Labels of the coordinate associated with one data dimension.
        Shape : ``(n_smpl,)``, where ``n_smpl`` should equal the length of the associated dimension
        in the data set.

    Methods
    -------
    :meth:`__init__`
    :meth:`__repr__`
    :meth:` __len__`
    :meth:`__eq__`
    :meth:`copy`
    :meth:`__getitem__`
    :meth:`build_labels` (abstract classmethod)
    :meth:`create`       (classmethod)
    :meth:`empty`        (classmethod)

    Notes
    -----
    Each coordinate is associated with a *single* dimension of the data set.
    Therefore, coordinate values are stored in 1D arrays (restriction compared to :mod:`xarray`).
    """

    def __init__(self, values: npt.NDArray[L]):
        self.values = values  # type: ignore[var-annotated]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> : {len(self)} samples."

    def __len__(self) -> int:
        """
        Number of coordinate labels, computed dynamically to accommodate any change in shape.

        Returns
        -------
        int
        """
        return self.values.size

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between two coordinates based on their underlying labels.

        Parameters
        ----------
        other: Any
            Object to compare with the coordinate.

        Returns
        -------
        bool
            True if all the elements in the numpy arrays are equal, False otherwise.
        """
        if not isinstance(other, type(self)):
            return False
        return np.array_equal(self.values, other.values)

    def copy(self) -> Self:
        """
        Copy the coordinate object with all its attributes.

        Returns
        -------
        Coordinate

        See Also
        --------
        :func:`copy.deepcopy`
        """
        return copy.deepcopy(self)

    @overload
    def __getitem__(self, idx: int) -> L: ...

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    @overload
    def __getitem__(self, idx: npt.NDArray[np.bool_]) -> Self: ...

    def __getitem__(self, idx: Union[int, slice, npt.NDArray[np.bool_]]) -> Union[L, Self]:
        """
        Index the coordinate labels by delegating to the underlying numpy array.

        Parameters
        ----------
        idx: Union[int, slice, npt.NDArray[bool]]
            Selection object accepted by numpy arrays: int, slice, boolean mask.

        Returns
        -------
        Union[L, Self]
            If the index is an integer, return the corresponding label.
            Otherwise, return a new coordinate object whose labels are restricted to the selected
            values. All the other attributes of the initial coordinate object are preserved.
        """
        if isinstance(idx, int):
            return self.values[idx]
        cpy = self.copy()
        cpy.values = self.values[idx]
        return cpy

    @classmethod
    @abstractmethod
    def build_labels(cls, *args, **kwargs) -> npt.NDArray:
        """
        Generate basic coordinate labels from minimal parameters, outside of full coordinate object.

        Parameters
        ----------
        *args, **kwargs
            Arguments required to build basic labels, specific to each coordinate subclass.

        Notes
        -----
        This *abstract* method requires actual implementations in subclasses, which may vary
        substantially depending on the type of coordinate and the parameters that they admit.
        In most coordinate sub-classes, a basic coordinate consists in an array filled with an
        arbitrary number of a unique label.

        Warnings
        --------
        In concrete implementations, the method's signature should *not* include variable and
        keyword arguments (*args, **kwargs). Indeed, the method is used in the method
        :meth:`create`, which applies the utility function :func:`filter_kwargs` to filter the
        arguments of :meth:`build_labels`. This can only retain the arguments which are *explicit*
        parameters of the method's signature.
        """

    @classmethod
    def create(cls: Type[C], **kwargs: Any) -> C:
        """
        Create a basic coordinate object from minimal parameters.

        Parameters
        ----------
        **kwargs: Any
            Arguments required by the constructor of the specific coordinate subclass, except the
            values of the coordinate since they are generated by the current method.
            Pass *named* arguments to ensure their correct assignment.

        Returns
        -------
        Coordinate
            Instance of the subclass with the generated labels.

        See Also
        --------
        :meth:`build_labels`
        :func:`filter_kwargs`
        """
        build_args = filter_kwargs(cls.build_labels, **kwargs)
        init_args = filter_kwargs(cls.__init__, **kwargs)
        values = cls.build_labels(**build_args)
        return cls(values=values, **init_args)

    @classmethod
    def empty(cls: Type[C]) -> C:
        """
        Create an empty coordinate object with no labels.

        Returns
        -------
        Coordinate
            Instance of the subclass with an empty array as labels.
        """
        return cls(values=np.empty(0))
