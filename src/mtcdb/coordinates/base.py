#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates.base` [module]

Abstract base class for data coordinates.

Classes
-------
:class:`Coordinate`
"""

from typing import Any, Type, TypeVar, Union

from abc import ABC, abstractmethod
import copy
import numpy as np
import numpy.typing as npt

from mtcdb.utils.functions import filter_kwargs


def display_counts():
    """TODO: Implement display_counts function."""

C = TypeVar('C', bound='Coordinate')
"""Generic type variable for Coordinate subclasses."""


class Coordinate(ABC):
    """
    Abstract Base Class representing labels for one dimension of a data set.

    This class is inherited in subclasses which represent specific types of coordinates.

    Attributes
    ----------
    values: npt.NDArray[Any]
        Labels for the data dimension.
        Shape : ``(n_smpl,)``, with ``n_smpl`` the number of coordinate samples.
        It equals the number of elements in the associated dimension
        of the data set.

    Methods
    -------
    :meth:`__init__`
    :meth:`__repr__`
    :meth:` __len__`
    :meth:`__eq__`
    :meth:`copy`
    :meth:`__getitem__`
    :meth:`build_labels` (classmethod, abstract)
    :meth:`create` (classmethod)

    Notes
    -----
    Each coordinate is associated with a *single* dimension of the data set.
    Therefore, coordinate values are stored in 1D arrays (restriction compared to ``xarray``).
    """
    def __init__(self, values: npt.NDArray[Any]):
        self.values = values

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> : {len(self)} samples."

    def __len__(self) -> int:
        """
        Number of coordinate labels (samples).
        
        Computed dynamically to accommodate any change of coordinate shape.
        
        Returns
        -------
        int
        """
        return self.values.size

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between two coordinates based on their values.

        Parameters
        ----------
        other: Any
            Object to compare with the coordinate.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if not isinstance(other, Coordinate):
            return False
        return np.array_equal(self.values, other.values)

    def copy(self: C) -> C:
        """
        Copy the coordinate object.

        Returns
        -------
        Coordinate

        See Also
        --------
        :func:`copy.deepcopy`
        """
        return copy.deepcopy(self)

    def __getitem__(self,
                    idx: Union[int, slice, npt.NDArray[np.bool_]]
                    ) -> Union['Coordinate', Any]:
        """
        Delegate the indexing to the coordinate values.

        Parameters
        ----------
        idx: Union[int, slice, npt.NDArray[bool]]
            Selection object accepted by numpy arrays 
            (int, slice, boolean mask, etc.).

        Returns
        -------
        Coordinate
            New coordinate object whose labels are the selected values.

        Warning
        -------
        The output is a new Coordinate object, not a simple array.
        All the other attributes of the original coordinate are preserved in the new one.
        """
        if isinstance(idx, int):
            return self.values[idx]
        cpy = self.copy()
        cpy.values = self.values[idx]
        return cpy

    @classmethod
    @abstractmethod
    def build_labels(cls, *args, **kwargs) -> npt.NDArray:
        """"
        Generate basic/default coordinates labels from a set of parameters.

        Notes
        -----
        This is an abstract method which requires actual implementations in
        subclasses, because different concrete implementations may vary substantially.
        Usually, this method creates an arbitrary number of a unique label.

        This method is static, so that is can be used flexibly
        without instantiating a coordinate object.
        Possible usages :
        
        - In processing pipelines, to build basic coordinates to compose in more complex ones.
        - In testing contexts, to get simple coordinates values without the full object.
        - In the method :meth:`create`, to provide values to the constructor.
        
        Warnings
        --------
        In concrete implementations, the method's signature should not include
        variable and keyword arguments (*args, **kwargs).
        Indeed, the method is used in the other method :meth:`create`,
        which applies the utility function :func:`filter_kwargs` to filter
        the arguments of :meth:`build_labels`. This only retains the arguments
        which are explicit parameters of the method's signature.
        """

    @classmethod
    def create(cls: Type[C], **kwargs: Any) -> C:
        """
        Create an instance of a subclass of Coordinate from scratch.

        Parameters
        ----------
        **kwargs: Any
            Specific arguments required to build coordinate labels.
            Pass *named* arguments to ensure their correct assignment.

        Returns
        -------
        Coordinate
            Instance of the subclass with the generated labels.

        Examples
        --------
        >>> CoordTime.create(n_smpl=10, t_max=1)
        <CoordTime> : 10 time points at 0.1 s bin.
        >>> CoordTask.create(n_smpl=10, cnd=Task.PTD)
        <CoordTask> : 10 samples, Task.PTD: 10.
        >>> CoordFold.create(n_smpl=10, k=5)
        <CoordFold> : 10 samples, [2, 2, 2, 2, 2].

        See Also
        --------
        :meth:`build_labels`
        :func:`filter_kwargs`
        """
        build_args = filter_kwargs(cls.build_labels, **kwargs)
        init_args = filter_kwargs(cls.__init__, **kwargs)
        values = cls.build_labels(**build_args)
        return cls(values=values, **init_args)
