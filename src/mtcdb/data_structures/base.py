#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.data_structures.base` [module]

Classes
-------
:class:`MetaData` (metaclass)
:class:`Data` (abstract base class, generic)

Implementation
--------------
Each subclass representing a data structure is build from a combination of
an metaclass and an abstract base class, with different responsibilities.

**MetaClass**

:class:`MetaData` serves to set *class-level* attributes for dimensions and coordinates.
In each subclass, only one attribute has to be defined: :attr:`dim2coord`.
Other class-level attributes are constructed from the latter by the metaclass:
:attr:`dims`, :attr:`coords` and :attr:`coord2dim`.

**Abstract Base Class**

:class:`Data` serves to define common methods and attributes for data structures.
Each subclass has to :
- Define its own dimensions and coordinates.
- Select its path manager, loader and saver subclasses (if needed).
- Implement its own constructor to define metadata required to build the path.
  In subclasses, the base class constructor should be called only when data is created, 
  not when it is loaded from a file.    
- Implement the abstract property :meth:`get_path` to provide the right arguments
  to its own path manager from its attributes.
- Override the abstract methods :meth:`load` and :meth:`save` (if needed) 
  to handle the specific data format. It should transform the data structure
  to the format expected by the loader or saver.
  By default, the data is loaded and saved with pickle.
  Thus, it is recovered directly as an instance of :class:`Data` 
  and can be saved directly without any transformation.

Warning
-------
Any transformation of the data which affects the dimensions should be prevented.
Namely: transpositions (dimension permutation), reshaping (dimension fusion)...
Several attributes are read-only and/or immutable to ensure the integrity of the data structure.
:attr:`data`
:attr:`dims`
:attr:`coords`
:attr:`shape`
:attr:`n`

Notes
-----
Since :class:`MetaData` is a subclass of :class:`abc.ABCMeta`,
it is not necessary to make :class:`Data` inherit from :class:`abc.ABC`.
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


T = TypeVar('T')
"""Type variable representing the type of data in the generic Data class."""

class MetaData(ABCMeta):
    """
    Metaclass for data structures. Define class-level attributes.

    Attributes
    ----------
    coords: Dict[str, FrozenSet[str]]
        Repertoire of coordinates associated to each dimension.
        Keys: Dimension name.
        Values: Coordinate names associated to this dimension.
                Each element of the container is one attribute under which 
                one coordinate is stored in the data structure.
    coord2dim: Mapping[str, str]
        Mapping of the coordinates to the dimensions of the data.
        Keys: Coordinate names.
        Values: Dimension names.
    
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
        if 'dim2coord' in dct: # dict of class attributes
            dct['dims'] = tuple(dct['dim2coord'].keys())
            dct['coord2dim'] = mcs.set_coord2dim(dct['dim2coord'])
            dct['coords'] = tuple(dct['coord2dim'].keys())
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def set_coord2dim(dim2coord: Dict[str, FrozenSet[str]]) -> Mapping[str, str]:
        """
        Map coordinates to dimensions.
        
        Returns
        -------
        Mapping[str, str]
            Mapping of the coordinates to the dimensions of the data.
            Keys: Coordinate names.
            Values: Dimension names.

        See Also
        --------
        :func:`mtcdb.utils.sequences.reverse_dict_container`
            Here, the output of the function on the class attribute :attr:`coords`
            is of the form {'coord': ['dim']},
            since each coordinate is associated with a single dimension. 
            Then, the list values are unpacked to extract the string values.
        """
        rev_dct: Dict[str, List[str]] = reverse_dict_container(dim2coord) # {'coord': ['dim']}}
        return {coord: dim[0] for coord, dim in rev_dct.items()}


class Data(Generic[T], metaclass=MetaData):
    """
    Abstract base class for data structures.
    
    Class Attributes
    ----------------
    dims: Tuple[str]
        Names of the dimensions (order matters).
    coords : Tuple[str]
        Names of the coordinates. 
        Each name specifies the attribute under which the coordinate is stored.
    dim2coord: Dict[str, FrozenSet[str]]
        Mapping from dimensions to their associated coordinates.
        Keys: Dimension names.
        Values: Coordinates associated to each dimension.
    coord2dim: Mapping[str, str]
        Mapping from coordinates to their associated dimensions.
        Keys: Coordinate names.
        Values: Dimension names.
    path_manager: Type[PathManager]
        Path manager class to build paths for data files.
    saver: Type[Saver], default=SaverPKL
        Saver class to save the data to files in a specific format.
    loader: Type[Loader], default=LoaderPKL
        Loader class to load the data from files.
    tpe : TargetType, default='object'
        Type of the loaded data (parameter for the loader).
    
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

    See Also
    --------
    :meth:`np.ndarray.setflags`: Used to make a numpy array immutable.
    :class:`MetaData`: Metaclass used to set class-level attributes.
    :class:`Generic`: Generic class to define a generic type.
    """
    dims: Tuple[str, ...] = ()
    coords: Tuple[str, ...] = ()
    coord2dim: Mapping[str, str] = MappingProxyType({})
    dim2coord: Mapping[str, FrozenSet[str]] = MappingProxyType({})
    path_manager: Type[PathManager]
    saver: Type[Saver] = SaverPKL
    loader: Type[Loader] = LoaderPKL
    tpe: TargetType = TargetType('object')

    def __init__(self, data: npt.NDArray, **kwargs) -> None:
        # Initialize data and dimensions
        self.data = data
        self.data.setflags(write=False) # make immutable
        self.shape = self.data.shape
        self.n = MappingProxyType({dim: shape for dim, shape in zip(self.dims, self.shape)})
        # Initialize coordinates
        for name in self.coords:
            if name not in kwargs:
                raise ValueError(f"Missing coordinate: {name}")
            setattr(self, name, kwargs[name])

    def __repr__(self) -> str:
        coord_names = list(self.coord2dim.keys())
        return f"<{self.__class__.__name__}> Dims: {self.dims}, Coords: {coord_names}"

    def copy(self) -> 'Data':
        """Copy the data structure."""
        return copy.deepcopy(self)

    @property
    @abstractmethod
    def path(self) -> Path:
        """Build the path to the file containing the data."""
        return self.path_manager().get_path()

    def load(self) -> 'Data':
        """Retrieve data from a file."""
        data = self.loader(path=self.path, tpe=self.tpe).load()
        return data

    def save(self) -> None:
        """Save the data instance."""
        self.saver(self.path, self.data).save()

    def sel(self, **kwargs) -> 'Data':
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
                dim = coord2dim[name] # dimension to which the coordinate applies
                coord = getattr(self, name) # coordinate object
                # Create a boolean mask depending on the target label type
                if isinstance(label, list):
                    mask = np.isin(coord.values, label)
                elif isinstance(label, slice):
                    mask = np.zeros(coord.values.size, dtype=bool)
                    mask[label] = True
                elif isinstance(label, (int, str, bool)):
                    mask = (coord.values == label) # pylint: disable=superfluous-parens
                else:
                    raise TypeError(f"Invalid type for label '{label}'.")
                # Apply this mask to the corresponding axis
                masks[dim] = masks[dim] & mask
        # Convert the boolean masks to integer indices
        indices = {dim: np.where(mask)[0] for dim, mask in masks.items()}
        # Combine masks along all dimensions (meshgrid)
        mesh = np.ix_(*[indices[dim] for dim in self.dims]) # order masks by dimensions
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
