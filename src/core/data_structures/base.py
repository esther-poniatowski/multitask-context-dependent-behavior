#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.data_structures.base` [module]

Classes
-------
:class:`DataStructMeta` (metaclass)
:class:`Data` (abstract base class, generic)
"""
from abc import ABCMeta, abstractmethod
import copy
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, List, Dict, FrozenSet, Mapping, Type, TypeVar, Generic, Optional, Self

import numpy as np
import numpy.typing as npt

from utils.storage_rulers.base import PathRuler
from utils.io_data.formats import TargetType
from utils.io_data.loaders.base import Loader
from utils.io_data.loaders.impl import LoaderDILL
from utils.io_data.savers.base import Saver
from utils.io_data.savers.impl import SaverDILL
from utils.misc.sequences import reverse_dict_container


class DataStructMeta(ABCMeta):
    """
    Metaclass to create data structures classes (subclasses of :class:`Data`).

    Responsibilities:

    - Ensure that the required class-level attributes are set in each subclass.
    - Set other class-level attributes to ensure their consistency with the provided values.

    Class Attributes
    ----------------
    required_attributes: List[str]
        Names of the class-level attributes which have to be defined in each subclass.

    Attributes
    ----------
    dim2coord: Data.dim2coord
    coord2type: Data.coord2type
    dims: Data.dims
    coords: Data.coords
    coord2dim: Data.coord2dim
    path_ruler: Data.path_ruler

    Methods
    -------
    :meth:`__new__`
    :meth:`check_class_attributes`
    :meth:`set_class_attributes`
    :meth:`reverse_mapping`

    See Also
    --------
    :class:`abc.ABCMeta`: Metaclass for *abstract base classes*.

    Warning
    -------
    In a metaclass, the instances are classes themselves (here, subclasses of :class:`Data`).
    """

    required_attributes = ["dim2coord", "coord2type", "path_ruler"]

    def __new__(mcs, name, bases, dct):
        """
        Create a subclass of :class:`Data` after ensuring its consistency.

        Parameters
        ----------
        dct: Dict[str, ...]
            Dictionary of class attributes. Initially, it contains the attributes defined in the
            body of the subclass being created. After the operations performed by the metaclass, it
            is updated with new attributes.

        Returns
        -------
        Type
            New class (subclass of :class:`Data`).

        See Also
        --------
        :meth:`check_class_attributes`
        :meth:`set_class_attributes`
        :meth:`check_consistency`
        :meth:`super().__new__`: Call the parent class constructor, here :class:`ABCMeta`.
        """
        mcs.check_class_attributes(dct)
        mcs.set_class_attributes(dct)
        mcs.check_consistency(dct)
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def check_class_attributes(dct):
        """
        Check that the required class-level attributes have been defined in the subclass body.

        Raise
        -----
        ValueError
            If any required attribute is missing in the subclass body.
        """
        for attr in DataStructMeta.required_attributes:
            if attr not in dct:
                raise ValueError(f"Missing attribute '{attr}'.")

    @staticmethod
    def set_class_attributes(dct):
        """Set the class-level attributes related to dimensions and coordinates."""
        dct["dims"] = tuple(dct["dim2coord"].keys())
        dct["coord2dim"] = DataStructMeta.reverse_mapping(dct["dim2coord"])
        dct["coords"] = tuple(dct["coord2dim"].keys())

    @staticmethod
    def check_consistency(dct):
        """
        Check that the class-level attributes are consistent with each other.

        Raise
        -----
        ValueError
            If the keys in :attr:`Data.coord2type` do not match :attr:`Data.coords`.
        """
        if set(dct["coord2type"].keys()) != set(dct["coords"]):
            raise ValueError("Inconsistent keys in 'coord2type'.")

    @staticmethod
    def reverse_mapping(dim2coord: Dict[str, FrozenSet[str]]) -> Mapping[str, str]:
        """
        Create the attribute :attr:`Data.coord2dim` by reversing the mapping :attr:`Data.dim2coord`.

        Parameters
        ----------
        dim2coord: Mapping[str, FrozenSet[str]]
            See :attr:`Data.dim2coord`.

        Returns
        -------
        coord2dim: Mapping[str, str]
            See :attr:`Data.coord2dim`.

        See Also
        --------
        :func:`utils.misc.sequences.reverse_dict_container`
            Here, the output of the function is of the form {'coord': ['dim']}, since each
            coordinate is associated with a single dimension.
            To obtain single string values instead of lists, each list is unpacked.
        """
        rev_dct: Dict[str, List[str]] = reverse_dict_container(dim2coord)  # {'coord': ['dim']}}
        return {coord: dim[0] for coord, dim in rev_dct.items()}


T = TypeVar("T")
"""Type variable representing the type of data in the generic Data class."""


class Data(Generic[T], metaclass=DataStructMeta):
    """
    Abstract base class for data structures, defining the interface to interact with data.

    Class Attributes
    ----------------
    dims: Tuple[str]
        Names of the dimensions (order matters).
    coords : Tuple[str]
        Names of each coordinate, specifying the attribute which stores the coordinate object.
    dim2coord: Dict[str, FrozenSet[str]]
        Mapping from dimensions to their associated coordinates.
        Keys: Dimension names.
        Values: Coordinates associated to each dimension.
    coord2dim: Mapping[str, str]
        Mapping from coordinates to their associated dimensions.
        Keys: Coordinate names.
        Values: Dimension names.
    coord2type: Mapping[str, Type]
        Mapping from coordinates to their types.
        Keys: Coordinate names.
        Values: Types of the coordinates, among the subclasses of :class:`Coordinate`.
    _has_data: bool
        Flag indicating if the data attribute has been filled with actual values.
    _has_coords: bool
        Flag indicating if the coordinates attributes have been filled with actual values.
    path_ruler: Type[PathRuler]
        Subclass of :class:`PathRuler` used to build paths to data files.
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
        Length of each dimension.
        Keys: Dimension names.
        Values: Number of elements.

    Methods
    -------
    :meth:`__init__`
    :meth:`set_data`
    :meth:`set_dims`
    :meth:`set_coords`
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
    immutable: :attr:`data`, :attr:`dims`, :attr:`coords`, :attr:`shape`, :attr:`n`.
    Moreover, for consistency, it is not recommended to transform the underlying numpy array through
    :mod:`numpy` functions such as transpositions (dimension permutation), reshaping (dimension
    fusion)...

    See Also
    --------
    :meth:`np.ndarray.setflags`: Used to make a numpy array immutable.
    :class:`DataStructMeta`: Metaclass used to set class-level attributes.
    :class:`Generic`: Generic class to define a generic type.
    :class:`Coordinate`: Base class for coordinates.
    :class:`PathRuler`: Base class for path managers.
    :class:`Loader`: Base class for loaders.
    :class:`Saver`: Base class for savers.
    :class:`TargetType`: Class to specify the type of the loaded data.

    Notes
    -----
    Since :class:`DataStructMeta` inherits from :class:`abc.ABCMeta`, it is not necessary to make
    :class:`Data` inherit from :class:`abc.ABC`.
    """

    # --- Dimensions and Coordinates ---
    # Required - Set in each subclass
    dim2coord: Mapping[str, FrozenSet[str]] = MappingProxyType({})
    coord2type: Mapping[str, Type] = MappingProxyType({})
    # Set by :class:`DataStructMeta` automatically
    dims: Tuple[str, ...] = ()
    coords: Tuple[str, ...] = ()
    coord2dim: Mapping[str, str] = MappingProxyType({})

    # --- IO Handlers ---
    # Required - Set in each subclass (here only declared)
    path_ruler = PathRuler
    # Optional - Overridden in some subclasses (here default values)
    saver: Type[Saver] = SaverDILL
    loader: Type[Loader] = LoaderDILL
    tpe: TargetType = TargetType("object")  # for :class:`LoaderPKL`

    def __init__(self, data: Optional[npt.NDArray], **kwargs) -> None:
        """
        Instantiate a data structure and check the consistency of the input values.

        Parameters
        ----------
        data:
            See :attr:`data`.
        kwargs: Dict[str, ...]
            Coordinates specific to the data structure subclass.
            Keys: Coordinate names as specified in :attr:`coord2type`.
            Values: Coordinate values, corresponding the the expected type of coordinate.

        See Also
        --------
        :meth:`set_data`
        :meth:`set_coords`
        :meth:`set_dims`
        """
        self._has_data = False
        self._has_coords = False
        # If provided, initialize actual values
        if data is not None:
            self.set_data(data)
            self.set_coords(**kwargs)
        # If not provided, initialize empty values of the expected types
        else:
            # Empty data array based on the dimensions
            shape = (0,) * len(self.dims)
            self.data = np.empty(shape=shape, dtype=np.float64)
            self.set_dims()
            # Empty coordinates based on the expected types
            for coord, cls in self.coord2type.items():
                setattr(self, coord, cls.empty())

    def set_data(self, data: npt.NDArray) -> None:
        """
        Set the attribute :attr:`data` with actual values.

        Parameters
        ----------
        data: npt.NDArray
            See :attr:`data`.

        Raises
        ------
        ValueError
            If the number of dimensions of the input data is not consistent with the dimensions
            expected for the data structure.

        See Also
        --------
        :meth:`np.ndarray.setflags`:
            Used to make a numpy array immutable to prevent any modification.
        :meth:`set_dims`:
            Used to update the attributes :attr:`shape` and :attr:`n` based on the current data
            values.
        :attr:`_has_data`:
            Set to True to indicate that the data attribute has been filled with actual values.
        """
        # Check the number of dimensions is consistency with :attr:`dims`
        if data.ndim != len(self.dims):
            raise ValueError(f"Invalid number of dimensions: {data.ndim} != {len(self.dims)}")
        self.data = data
        self.data.setflags(write=False)  # make immutable
        self.set_dims()  # update the dimensions based on the data
        self._has_data = True

    def set_dims(self):
        """Set the attributes :attr:`shape` and :attr:`n` based on :attr:`data`."""
        self.shape = self.data.shape  # delegate to numpy array
        self.n = MappingProxyType({dim: shape for dim, shape in zip(self.dims, self.shape)})

    def set_coords(self, **kwargs) -> None:
        """
        Set the coordinate attributes and check their consistency.

        Parameters
        ----------
        kwargs: Dict[str, ...]
            See :meth:`__init__`.

        Raises
        ------
        ValueError
            If any required coordinate is missing in the keyword arguments.
        TypeError
            If the argument passed for one coordinate does not match its expected type.
        ValueError
            If the length of the argument passed for one coordinate does not match the length of its
            associated data axis.

        See Also
        --------
        :attr:`coord2type`:
            Used to check the presence and type of the coordinate values.
        :attr:`n`:
            Has to been set beforehand by the method :meth:`set_dims`.
        :attr:`Coordinate.__len__`
             assumed to be implemented in the coordinate classes.
        :attr:`_has_coords`:
            Set to True to indicate that the coordinate attributes have been filled with actual
            values.
        """
        for coord, tpe in self.coord2type.items():
            # Check coordinate presence
            if coord not in kwargs:
                raise ValueError(f"Missing coordinate: {coord}")
            # Check coordinate type
            if not isinstance(kwargs[coord], tpe):
                raise TypeError(f"Invalid type for coordinate '{coord}': {type(kwargs[coord])}")
            # Check coordinate length
            if len(kwargs[coord]) != self.n[self.coord2dim[coord]]:
                raise ValueError(f"Invalid length for coordinate '{coord}': {len(kwargs[coord])}")
            setattr(self, coord, kwargs[coord])
        self._has_coords = True

    def __repr__(self) -> str:
        coord_names = list(self.coord2dim.keys())
        return f"<{self.__class__.__name__}> Dims: {self.dims}, Coords: {coord_names}"

    def copy(self) -> Self:
        """Return a *deep copy* of the data structure."""
        return copy.deepcopy(self)

    @property
    @abstractmethod
    def path(self) -> Path:
        """Abstract Property - Build the path to the file containing the data."""
        return self.path_ruler().get_path()  # type: ignore[abstract] # pylint: disable=abstract-class-instantiated

    def load(self) -> None:
        """Retrieve an instance from the file at :attr:`path`."""
        loaded_obj = self.loader(path=self.path, tpe=self.tpe).load()
        self.__dict__.update(loaded_obj.__dict__)

    def save(self) -> None:
        """Save an instance to a file at :attr:`path` in the format specific to the saver."""
        self.saver(self.path, self).save()

    def __getitem__(self, coord_name: str):
        """
        Retrieve a coordinate using its name as a string.

        Parameters
        ----------
        coord_name: str
            Name of the coordinate to retrieve.

        Returns
        -------
        Coordinate
            Coordinate object stored in the data structure.

        Raises
        ------
        KeyError
            If the coordinate name is not found in the data structure.

        Example
        -------
        Access the coordinate 'time':
        >>> data['time']
        """
        if coord_name in self.coord2dim:
            return getattr(self, coord_name)
        else:
            raise KeyError(f"Coordinate '{coord_name}' not found in the data structure.")

    def sel(self, **kwargs) -> Self:
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
