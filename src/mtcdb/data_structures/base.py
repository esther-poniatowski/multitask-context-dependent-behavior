#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.data_structures.base` [module]

Classes
-------
:class:`Data` (abstract base class, generic)
"""
from abc import ABC, abstractmethod
import copy
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, Dict, Mapping, Type, TypeVar, Generic

import numpy as np
import numpy.typing as npt

from mtcdb.io_handlers.path_managers.base import PathManager
from mtcdb.io_handlers.formats import TargetType
from mtcdb.io_handlers.loaders.base import Loader
from mtcdb.io_handlers.loaders.impl import LoaderPKL
from mtcdb.io_handlers.savers.base import Saver
from mtcdb.io_handlers.savers.impl import SaverPKL


T = TypeVar('T')
"""Type variable representing the type of data in the generic Data class."""


class Data(ABC, Generic[T]):
    """
    Base class for data structures.
    
    Class Attributes
    ----------------
    dims: Tuple[str]
        Names of the dimensions.
        Name of an axis: ``dims[axis]``
        Axis of a dimension: ``dims.index(name)``
    coords: Dict[str, str]
        Repertoire of coordinates associated to each dimension.
        Keys: Dimension name.
        Values: Coordinate name, i.e. attribute under which 
                it is stored in the data structure.
    path_manager: Type[PathManager]
        Path manager class to build paths for data files.
        (In each data subclass, pick the right path manager subclass).
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
    coord2dim: Mapping[str, str]
        Mapping of the coordinates to the dimensions of the data.
        Keys: Coordinate names.
        Values: Dimension names.
        
    Methods
    -------
    :meth:`__init__`
    :meth:`__repr__`
    :meth:`copy`
    :meth:`path`
    :meth:`load`
    :meth:`save`
    :meth:`coord2dim`
        
    
    Examples
    --------
    Access the number of time points in data with a time dimension:
    >>> data.n['time']

    Notes
    -----
    Several attributes are read-only and/or immutable to ensure the integrity of the data structure.
    :attr:`data`
    :attr:`dims`
    :attr:`coords`
    :attr:`shape`
    :attr:`n`

    Warning
    -------
    Any transformation of the data which affects the dimensions should be prevented.
    Namely : transpositions (dimension permutation), reshaping (dimension fusion).

    Implementation
    --------------
    Information associated to each dimension is specified in dictionaries
    whose keys are dimensions names (instead of tuples in which the order matters).
    This is relevant to to abstract away from the order of the dimensions
    which might change in data transformations.
    
    See Also
    --------
    :meth:`npt.NDArray.setflags`
        Used to make a numpy array read-only.
    """
    dims: Tuple[str]
    coords: Dict[str, str]
    path_manager: Type[PathManager]
    saver: Type[Saver] = SaverPKL
    loader: Type[Loader] = LoaderPKL
    tpe: TargetType = TargetType('object')

    def __init__(self, data: npt.NDArray) -> None:
        self.data = data
        self.data.setflags(write=False) # make immutable
        self.shape = self.data.shape
        self.n = MappingProxyType({dim: shape for dim, shape in zip(self.dims, self.shape)})

    def __repr__(self) -> str:
        coord_names = list(self.coords.values())
        return f"<{self.__class__.__name__}> Dims: {self.dims}, Coords: {coord_names}"

    @classmethod
    def coord2dim(cls) -> Mapping[str, str]:
        """
        Map coordinates to dimensions.
        
        Returns
        -------
        Mapping[str, str]
            Mapping of the coordinates to the dimensions of the data.
            Keys: Coordinate names.
            Values: Dimension names.

        Notes
        -----
        This is a method rather than an immutable attribute
        because it should be specific to each subclass.
        """
        return MappingProxyType({coord: dim for dim, coord in cls.coords.items()})

    def copy(self) -> 'Data':
        """
        Copy the data structure.

        Returns
        -------
        Data

        See Also
        --------
        :meth:`copy.deepcopy`
        """
        return copy.deepcopy(self)

    @property
    @abstractmethod
    def path(self) -> Path:
        """
        Build the path to the file containing the data.

        Overridden in data subclasses to provide the required arguments 
        to the path manager from the attributes of the data structure.

        Returns
        -------
        Path
        """
        return self.path_manager().get_path()

    def load(self) -> 'Data':
        """
        Retrieve data from a file.

        Notes
        -----
        The raw data is recovered in the type specified by :obj:`tpe`.
        If needed, transform it in an instance of the data structure.
        By default, with pickle, the data is directly recovered 
        as an object corresponding to the data structure.

        Returns
        -------
        Data
        """
        data = self.loader(path=self.path, tpe=self.tpe).load()
        data = self.__class__(data) # call constructor
        return data

    def save(self) -> None:
        """
        Save the data instance.

        Notes
        -----
        If needed, the data structure should be transformed
        in the format expected by the saver.
        By default, with pickle, the object can be saved directly
        without any transformation.
        """
        self.saver(self.path, self.data).save()

    @property
    def coord_attrs(self) -> Tuple[str]:
        """TODO"""
        raise NotImplementedError("Method not implemented yet.")

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
        coord2dim = self.coord2dim()
        coord_attrs = list(coord2dim.keys())
        # Initialize True masks for each dimensions to select all elements by default
        masks = {dim: np.ones(shape, dtype=bool) for dim, shape in zip(self.dims, self.shape)}
        # Update masks with the selection criteria on each coordinate
        for name, label in kwargs.items():
            if name in self.coord_attrs:
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
            if name in self.coord_attrs:
                new_coords[name] = coord[indices[coord2dim[name]]]
        # Instantiate a new data structure with the selected data,
        # by unpacking the new coordinates dictionary
        raise NotImplementedError("Method not implemented yet.")
        return self.__class__(new_data, **new_coords)
