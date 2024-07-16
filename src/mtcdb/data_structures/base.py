#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.data_structures.base` [module]

Classes
-------
:class:`MetaData` (metaclass) :class:`Data` (abstract base class, generic)

Implementation
--------------
Each subclass representing a data structure is build from a combination of a metaclass and an
abstract base class, with different responsibilities.

- :class:`MetaData` defines *class-level* attributes related to the dimensions and coordinates of
  the data structure.
- :class:`Data` defines methods and attributes shared by the various concrete data structures
  subclasses.

**Defining Concrete Data Structures**

Each data structure subclass should:

- Define its intrinsic dimensions and coordinates (class-level attributes). Only one attribute has
  to be specified: :attr:`dim2coord`. The other class-level attributes are constructed from the
  latter by the metaclass: :attr:`dims`, :attr:`coords` and :attr:`coord2dim`.
- Select its path manager among the subclasses of :class:`PathManager`, by setting the class-level
  attribute :attr:`path_manager`.
- #TODO Implement its own constructor to provide two instantiation approaches: creating of new data
  from scratch or loading existing data from a file. For both, it should set the *minimal metadata*
  required to build the path. For new data creation, it should admit data values and coordinates as
  optional parameters and call the base class constructor.
- Implement the abstract property :meth:`get_path` to provide the right arguments to its own path
  manager from its attributes.
- Change the loader and the saver (optional) and override the abstract methods :meth:`load` and
  :meth:`save` to transform the data structure into the format expected by the loader or saver. By
  default, the data is handled with :mod:`pickle`, which allows to recover it as an instance of
  :class:`Data` and to save it directly without any transformation.
"""
from abc import ABCMeta, abstractmethod
import copy
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, List, Dict, FrozenSet, Mapping, Type, TypeVar, Generic

import numpy as np
import numpy.typing as npt

from mtcdb.io_handlers.path_managers.base import PathManager
from mtcdb.io_handlers.formats import TargetType
from mtcdb.io_handlers.loaders.base import Loader
from mtcdb.io_handlers.loaders.impl import LoaderPKL
from mtcdb.io_handlers.savers.base import Saver
from mtcdb.io_handlers.savers.impl import SaverPKL
from mtcdb.utils.sequences import reverse_dict_container


T = TypeVar("T")
"""Type variable representing the type of data in the generic Data class."""


class MetaData(ABCMeta):
    """
    Metaclass for data structures. Define class-level attributes for dimensions and coordinates.

    Attributes
    ----------
    dims: Data.dims
    coords: Data.coords
    coord2dim: Data.coord2dim

    Methods
    -------
    :meth:`__new__`
    :meth:`set_coord2dim`

    See Also
    --------
    :class:`abc.ABCMeta`: Metaclass for abstract base classes.
    """

    def __new__(mcs, name, bases, dct):
        """Set class attributes for dimensions and coordinates from :attr:`dim2coord`."""
        if "dim2coord" in dct:  # `dct`: dictionary of class attributes
            dct["dims"] = tuple(dct["dim2coord"].keys())
            dct["coord2dim"] = mcs.set_coord2dim(dct["dim2coord"])
            dct["coords"] = tuple(dct["coord2dim"].keys())
        else:
            raise ValueError("Missing attribute 'dim2coord'.")
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def set_coord2dim(dim2coord: Dict[str, FrozenSet[str]]) -> Mapping[str, str]:
        """
        Map coordinates to dimensions.

        Returns
        -------
        Mapping[str, str]

        See Also
        --------
        :func:`mtcdb.utils.sequences.reverse_dict_container`
            Here, the output of the function is of the form {'coord': ['dim']}, since each
            coordinate is associated with a single dimension. Each list is unpacked to extract the
            single string value.
        """
        rev_dct: Dict[str, List[str]] = reverse_dict_container(dim2coord)  # {'coord': ['dim']}}
        return {coord: dim[0] for coord, dim in rev_dct.items()}


class Data(Generic[T], metaclass=MetaData):
    """
    Abstract base class for data structures.

    Class Attributes
    ----------------
    dims: Tuple[str]
        Names of the dimensions (order matters).
    coords : Tuple[str]
        Names of the coordinates. Each name specifies the attribute which stores the coordinate.
    dim2coord: Dict[str, FrozenSet[str]]
        Mapping from dimensions to their associated coordinates.
        Keys: Dimension names.
        Values: Coordinates associated to each dimension.
    coord2dim: Mapping[str, str]
        Mapping from coordinates to their associated dimensions.
        Keys: Coordinate names.
        Values: Dimension names.
    path_manager: Type[PathManager]
        Subclass of :class:`PathManager` used to build paths to data files.
    saver: Type[Saver], default=SaverPKL
        Subclass of :class:`Saver` used to save data to files in a specific format.
    loader: Type[Loader], default=LoaderPKL
        Subclass of :class:`Loader` used to load data from files in a specific format.
    tpe : TargetType, default='object'
        Type of the loaded data (parameter for the method :meth:`Loader.load`).

    Attributes
    ----------
    data: npt.NDArray
        Actual data values to analyze.
    shape: Tuple[int]
        Shape of the data array (delegated to the numpy array).
    n: MappingProxyType[str, int]
        Number of elements along each dimension.
        Keys: Dimension names.
        Values: Number of elements.

    Methods
    -------
    :meth:`__init__`
    :meth:`__repr__`
    :meth:`copy`
    :meth:`path`
    :meth:`load`
    :meth:`save`

    Examples
    --------
    Access the number of time points in data with a time dimension:
    >>> data.n['time']

    Get the name of an axis:
    >>> data.dims[0]

    Get the axis of the time dimension:
    >>> dims.index('time')

    Warning
    -------
    The dimensions and types of coordinates are *intrinsic* to each data structure, they are part of
    its core property.
    To ensure the integrity of the data structure, several attributes are read-only and/or
    immutable: :attr:`data`, :attr:`dims`, :attr:`coords`, :attr:`shape`, :attr:`n`...
    Moreover, for consistency, it is not recommended to transform the underlying numpy array through
    :mod:`numpy` functions such as transpositions (dimension permutation), reshaping (dimension
    fusion)...

    See Also
    --------
    :meth:`np.ndarray.setflags`: Used to make a numpy array immutable.
    :class:`MetaData`: Metaclass used to set class-level attributes.
    :class:`Generic`: Generic class to define a generic type.

    Notes
    -----
    Since :class:`MetaData` is a subclass of :class:`abc.ABCMeta`, it is not necessary to make
    :class:`Data` inherit from :class:`abc.ABC`.
    """

    dims: Tuple[str, ...] = ()  # set by :class:`MetaData`
    coords: Tuple[str, ...] = ()  # set by :class:`MetaData`
    coord2dim: Mapping[str, str] = MappingProxyType({})  # set by :class:`MetaData`
    dim2coord: Mapping[str, FrozenSet[str]] = MappingProxyType({})  # required by :class:`MetaData`
    path_manager: Type[PathManager]  # to be set by subclasses
    saver: Type[Saver] = SaverPKL  # default
    loader: Type[Loader] = LoaderPKL  # default
    tpe: TargetType = TargetType("object")  # for :class:`LoaderPKL`

    def __init__(self, data: npt.NDArray, **kwargs) -> None:
        # Initialize data and dimensions
        self.data = data
        self.data.setflags(write=False)  # make immutable
        self.shape = self.data.shape  # delegate to numpy array
        self.n = MappingProxyType({dim: shape for dim, shape in zip(self.dims, self.shape)})
        # Initialize coordinates
        for name in self.coords:
            if name not in kwargs:
                raise ValueError(f"Missing coordinate: {name}")
            setattr(self, name, kwargs[name])

    def __repr__(self) -> str:
        coord_names = list(self.coord2dim.keys())
        return f"<{self.__class__.__name__}> Dims: {self.dims}, Coords: {coord_names}"

    def copy(self) -> "Data":
        return copy.deepcopy(self)

    @property
    @abstractmethod
    def path(self) -> Path:
        """Abstract Property - Build the path to the file containing the data."""
        return self.path_manager().get_path()

    def load(self) -> "Data":
        """Retrieve data from a file at the path in :attr:`path`."""
        data = self.loader(path=self.path, tpe=self.tpe).load()
        return data

    def save(self) -> None:
        """Save data to a file at the path in :attr:`path` in a format specific to the saver."""
        self.saver(self.path, self.data).save()

    def sel(self, **kwargs) -> "Data":
        """
        Select data along specific coordinates.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            Keys : Coordinate names.
            Values : Selection criteria (single value, list or slice).

        Returns
        -------
        Data
            New data structure containing the selected data.

        Example
        -------
        Select time points between 0 and 1 second:
        >>> data.sel(time=slice(0, 1))
        Select trials in task 'PTD':
        >>> data.sel(task='PTD')
        Select trials for stimuli 'R' and 'T':
        >>> data.sel(stim=['R', 'T'])
        Select error trials only:
        >>> data.sel(error=True)
        Select along multiple coordinates:
        >>> data.sel(time=slice(0, 1), task='PTD', stim=['R', 'T'])
        """
        # Unpack the names of the coordinates present in the data structure
        coord2dim = self.coord2dim
        coord_attrs = list(coord2dim.keys())
        # Initialize True masks for each dimensions to select all elements by default
        masks = {dim: np.ones(shape, dtype=bool) for dim, shape in zip(self.dims, self.shape)}
        # Update masks with the selection criteria on each coordinate
        for name, label in kwargs.items():
            if name in coord_attrs:
                dim = coord2dim[name]  # dimension to which the coordinate applies
                coord = getattr(self, name)  # coordinate object
                # Create a boolean mask depending on the target label type
                if isinstance(label, list):
                    mask = np.isin(coord.values, label)
                elif isinstance(label, slice):
                    mask = np.zeros(coord.values.size, dtype=bool)
                    mask[label] = True
                elif isinstance(label, (int, str, bool)):
                    mask = coord.values == label  # pylint: disable=superfluous-parens
                else:
                    raise TypeError(f"Invalid type for label '{label}'.")
                # Apply this mask to the corresponding axis
                masks[dim] = masks[dim] & mask
        # Convert the boolean masks to integer indices
        indices = {dim: np.where(mask)[0] for dim, mask in masks.items()}
        # Combine masks along all dimensions (meshgrid)
        mesh = np.ix_(*[indices[dim] for dim in self.dims])  # order masks by dimensions
        # Select the data
        new_data = self.data[mesh]
        # Select the coordinates
        new_coords = {name: getattr(self, name) for name in coord_attrs}
        for name, _ in kwargs.items():
            if name in coord_attrs:
                new_coords[name] = coord[indices[coord2dim[name]]]
        # Instantiate a new data structure with the selected data,
        # by unpacking the new coordinates dictionary
        raise NotImplementedError("Method not implemented yet.")
        # return self.__class__(new_data, **new_coords)
