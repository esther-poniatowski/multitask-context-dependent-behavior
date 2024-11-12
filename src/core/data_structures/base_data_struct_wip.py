#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.base_data_struct` [module]


"""
from abc import ABC
import copy
from pathlib import Path
from typing import Tuple, Mapping, Type, TypeVar, Generic, Optional, Self, Union

import numpy as np

from core.data_structures.core_data import CoreData, Dimensions

# from core.coordinates.base_coord import Coordinate
from utils.storage_rulers.base_path_ruler import PathRuler
from utils.io_data.base_loader import Loader
from utils.io_data.loaders import LoaderDILL
from utils.io_data.base_saver import Saver
from utils.io_data.savers import SaverDILL


class Coordinate(np.ndarray):
    """TODO: Refactor the Coordinate class in ``core.coordinates.base_coord`` to inherit from
    np.ndarray"""


T = TypeVar("T")
"""Type variable representing the type of data in the generic Data class."""

C = TypeVar("C", bound=Union[CoreData, Coordinate])
"""Type variable representing the type of content attributes (CoreData, Coordinate)."""


# TODO: Could I simplify this class ?
# TODO: Should I convert the class attributes to uppercase to distinguish them from instance
# attributes ? Then I should delegate the access to the lowercase dims.
# TODO: Is hasattr raising an error is the attribute is declared but not set ?
# TODO: Should I define a type guard for setting coordinates and checking consistency?
# TODO: Is the path and loader implemented in a relevant way ?


class DataStructure(Generic[T], ABC):
    """ """

    # --- Class-Level Configurations ---------------------------------------------------------------

    # --- Schema of the Data Structure ---
    _REQUIRED_ATTRIBUTES = ("dims", "coords", "coords_to_dims", "identifiers")
    dims: Dimensions
    coords: Mapping[str, type]
    coords_to_dims: Mapping[str, Dimensions]
    identifiers: Tuple[str, ...]

    def __init_subclass__(cls) -> None:
        """ """
        # Call parent hook (default behavior)
        super().__init_subclass__()
        # Check class-level attributes
        for class_attr in cls._REQUIRED_ATTRIBUTES:
            if not hasattr(cls, class_attr):
                raise TypeError(f"<{cls.__name__}> Missing class-level attribute: '{class_attr}'.")
        for coord_name, coord_dims in cls.coords_to_dims.items():
            for dim in coord_dims:
                if dim not in cls.dims:
                    raise AttributeError(f"Invalid dimension '{dim}' for coord '{coord_name}'.")

    # --- Instance-Level Manipulations -------------------------------------------------------------

    def __init__(
        self,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        **coords_args: Optional[Union[Coordinate, np.ndarray]],
    ) -> None:
        """ """
        # Lazy initialization: declare private and empty content-related attributes
        self.data: CoreData
        # Fill with actual values if provided
        if data is not None:
            self.set_data(data)
        for coord_name, coord_value in coords_args.items():
            if coord_value is not None:  # pass individual coordinates to the setter method
                self.set_coords(**{coord_name: coord_value})

    def get_coord(self, coord_name: str) -> Coordinate:
        """
        Retrieve a coordinate using its attribute name. Useful to iterate over coordinates.

        Parameters
        ----------
        coord_name : str
            Name of the coordinate to retrieve.

        Returns
        -------
        coord_values : Coordinate
            Coordinate object stored in the data structure.
        """
        if coord_name not in self.coords:
            raise AttributeError(f"Invalid coordinate: '{coord_name}' not in {self.coords.keys()}.")
        if not hasattr(self, coord_name):
            raise AttributeError(f"Coordinate '{coord_name}' not set.")
        return getattr(self, coord_name)

    def __getattr__(self, name):
        """
        Delegate the access to nested attributes of the coordinate objects.

        Parameters
        ----------
        name : str
            Name of the attribute to get.
        """
        if name == "get_dim":
            return self.dims.get_dim
        elif name == "get_axis":
            return self.dims.get_axis
        elif name == "get_size":
            return self.data.get_size
        elif name == "shape":
            return self.data.shape
        else:
            for obj in self.__dict__.values():
                if hasattr(obj, name):
                    return getattr(obj, name)
            raise AttributeError(
                f"Invalid attribute '{name}' for object '{self.__class__.__name__}'."
            )

    # --- Set Data and Coordinates -----------------------------------------------------------------

    def set_data(self, data: Union[CoreData, np.ndarray]) -> None:
        """
        Set the attribute `data` with actual values and check its consistency.
        """
        if data.ndim != len(self.dims):
            raise ValueError(f"Invalid number of dimensions: data {data.ndim} != {len(self.dims)}")
        if not isinstance(data, CoreData):  # convert to CoreData with the expected dims
            data = CoreData(data, self.dims)
        for coord_name in self.coords:
            if hasattr(self, coord_name):
                self.check_consistency(data, coord_name, self.get_coord(coord_name))
        setattr(self, "data", data)

    def set_coords(self, **coords_args: Union[Coordinate, np.ndarray]) -> None:
        """
        Set coordinate attributes (all or a subset) and check their consistency.
        """
        for name, value in coords_args.items():
            if name not in self.coords:
                raise ValueError(f"Invalid coordinate name: '{name}'.")
            if not isinstance(value, self.coords[name]):  # convert to the expected coordinate type
                value = self.coords[name](value)
            if hasattr(self, "data"):  # check consistency with the data structure dimensions
                self.check_consistency(self.data, name, value)
            setattr(self, name, value)

    def check_consistency(self, data: CoreData, coord_name: str, coord_value: Coordinate):
        """
        Check consistency between the shape of the data and its coordinates.
        """
        data_shape = tuple(data.get_size(dim) for dim in self.coords_to_dims[coord_name])
        if coord_value.shape != data_shape:
            raise ValueError(
                f"Invalid shape: {coord_value.shape} (coord '{coord_name}') != {data_shape} (data)"
            )


from types import MappingProxyType


class TestClass(DataStructure):

    dims = Dimensions("trials", "time")
    coords = MappingProxyType({"time": Coordinate, "task": Coordinate})
    coords_to_dims = MappingProxyType({"time": Dimensions("time"), "task": Dimensions("trials")})
    identifiers = ("id_",)

    def __init__(
        self,
        id_: str,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        time: Optional[Union[Coordinate, np.ndarray]] = None,
        task: Optional[Union[Coordinate, np.ndarray]] = None,
    ):
        super().__init__(data=data, time=time, task=task)
        self.id_ = id_

    def test_method(self):
        """test"""
        print(self.data.shape)
        print(self.time.shape)
        print(self.get_coord("time").shape)
