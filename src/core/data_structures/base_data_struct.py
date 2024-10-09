#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.base_data_struct` [module]

Classes
-------
`LazyAttribute`
`DataStructure` (abstract base class, generic)

Notes
-----
Lazy Initialization of Content-Related Attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Content-related attributes (data and coordinates) are initialized as empty and can be set after the
initialization of the data structure.

Interactions with those attributes are handled by a dedicated descriptor. Roles of the descriptor:

- For the client code, it raises an explicit error message and prevents using the data and
  coordinates attributes *before* they are populated with actual values.
- For type checkers, it eliminates warning messages raised when the attributes are manipulated
  within the data structure methods (since they are declared as `Optional`).

Separation of concerns for interacting with content-related attributes:

- The `__subclass_init__` method of the data structure base class declares the content-related
  attributes as instances of the lazy descriptor.
- The constructor of the data structure class initializes *private* attributes with empty values
  (`None`).
- The `LazyAttribute` descriptor handles the interactions with to the content-related attributes
  when their *public* name is used. The descriptor is involved in most of the data structure's
  methods, which only *access* to the attributes through a unified interface.
- Two dedicated setter methods in the data structure class (`set_data` and `set_coords`) allow
  setting actual content in the attributes after consistency validation. Those methods use *private*
  names (e.g., `_data`, `_coord_name`...) to bypass the descriptor.
- The `__repr__` method uses the *private* names of the attributes to check their state (empty or
  filled).

Note: Moving the validation logic to the descriptor would make it complex and tightly coupled to
specific data validation rules.

Separation of concerns between the base and subclasses constructors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Base class constructor: Declare the content-related attributes (data and coordinates) with their
  respective types, and initialize them with values if provided.
- Subclass constructors: Assign the metadata attributes (identifiers for the data structure and
  optional descriptive information) AND call the base class constructor with the content-related
  arguments.

"""
from abc import ABC
import copy
from pathlib import Path
from typing import Tuple, Mapping, Type, TypeVar, Generic, Optional, Self, Union

import numpy as np

from core.data_structures.core_data import CoreData, Dimensions

# from core.coordinates.base_coord import Coordinate
from utils.storage_rulers.base_path_ruler import PathRuler
from utils.io_data.formats import TargetType
from utils.io_data.loaders.base_loader import Loader
from utils.io_data.loaders.impl_loaders import LoaderDILL
from utils.io_data.savers.base_saver import Saver
from utils.io_data.savers.impl_savers import SaverDILL


class Coordinate(np.ndarray):
    """TODO: Refactor the Coordinate class in ``core.coordinates.base_coord`` to inherit from
    np.ndarray"""


T = TypeVar("T")
"""Type variable representing the type of data in the generic Data class."""

C = TypeVar("C", bound=Union[CoreData, Coordinate])
"""Type variable representing the type of content attributes (CoreData, Coordinate)."""


class LazyAttribute(Generic[C]):
    """
    Descriptor for lazy initialization of data and coordinates in the data structure.

    Attributes
    ----------
    name: str
        Name of the lazy attribute to handle.

    Parameters
    ----------
    name: str
        See the descriptor attribute `LazyAttribute.name`.
    instance: DataStructure
        Instance of the data structure which stores the attribute.
    owner: Type[DataStructure]
        Class of the data structure.

    Raises
    ------
    AttributeError
        If the lazy attribute is accessed before being populated with actual values.
    """

    def __init__(self, name):
        self.name = f"_{name}"  # private attribute name

    def __get__(self, instance: "DataStructure", owner: Type["DataStructure"]) -> C:
        value = getattr(instance, self.name)
        if value is None:
            raise AttributeError(f"Unset attribute: '{self.name}'.")
        return value


class DataStructure(Generic[T], ABC):
    """
    Abstract base class for data structures, defining the interface to interact with data.

    Class Attributes
    ----------------
    _REQUIRED_ATTRIBUTES : Tuple[str]
        Names of the class-level attributes which have to be defined in each data structure
        subclass.
    dims : Dimensions
        Names of the dimensions, ordered to label the axes of the underlying data.
    coords : Mapping[str, type]
        Names of the coordinates (as attributes) and their expected types.
    coords_to_dims : Mapping[str, Tuple[str, ...]]
        Mapping from coordinates to their associated dimension(s) in the data structure.
        Keys: Coordinate names.
        Values: Dimension names, ordered to match the coordinate axes.
    identifiers : Tuple[str, ...]
        Names of the metadata attributes which jointly and uniquely identify each data structure
        instance within its class. Handled by each subclass' constructor.
    path_ruler : Type[PathRuler]
        Subclass of `PathRuler` used to build the path to file where the content of the data
        structure instance can be saved to and/or loaded from.
    saver : Type[Saver], default=SaverDILL
        Subclass of `Saver` used to save data to a file in a specific format.
    loader : Type[Loader], default=LoaderDILL
        Subclass of `Loader` used to load data from a file in a specific format.
    tpe : TargetType, default='object'
        Type of the loaded data (parameter for the method `Loader.load`).

    Attributes
    ----------
    data: CoreData
        Actual data values to analyze.
    shape: Tuple[int]
        (Property) Shape of the data array (delegated to the numpy array).
    path: Path
        (Property) Path to the file where the data can be stored.

    Methods
    -------
    `get_coord`
    `__getattr__`
    `set_data`
    `set_coords`
    `load`
    `save`
    `copy`
    `sel`

    Examples
    --------
    Get the name of an axis (two equivalent approaches):

    >>> data.dims[0]
    "time"
    >>> data.get_dim(0) # delegated to the core data object
    "time"

    Get the axis of the time dimension:

    >>> data.get_axis('time') # delegated to the core data object
    0

    Get the size of the time dimension:

    >>> data.get_size('time') # delegated to the core data object
    10

    Get the time coordinate (two equivalent approaches):

    >>> data.time
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> data.get_coord('time')
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Warning
    -------
    The dimensions and types of coordinates are *intrinsic* properties of each data structure class.
    To ensure the integrity of the data structure, it is not recommended to transform the underlying
    numpy array through `numpy` functions such as transpositions (dimension permutation),
    reshaping (dimension fusion)...

    See Also
    --------
    :class:`Generic`: Generic class to define a generic type.
    :class:`ABC`: Abstract base class to define abstract methods and properties.
    :class:`Coordinate`: Base class for coordinates.
    :class:`PathRuler`: Base class for path managers.
    :class:`Loader`: Base class for loaders.
    :class:`Saver`: Base class for savers.
    :class:`TargetType`: Class to specify the type of the loaded data.
    """

    # --- Class-Level Configurations ---------------------------------------------------------------

    # --- Schema of the Data Structure ---
    _REQUIRED_ATTRIBUTES = ("dims", "coords", "coords_to_dims", "identifiers")
    dims: Dimensions
    coords: Mapping[str, type]
    coords_to_dims: Mapping[str, Dimensions]
    identifiers: Tuple[str, ...]

    # --- IO Handlers ---
    # Optional - Overridden in subclasses if necessary (here default values)
    path_ruler: Type[PathRuler]
    saver: Type[Saver] = SaverDILL
    loader: Type[Loader] = LoaderDILL
    tpe: TargetType = TargetType("object")  # for :class:`LoaderPKL`

    def __init_subclass__(cls) -> None:
        """
        Hook method called when any subclass of `DataStructure` is created.

        Parameters
        ----------
        cls : Type[DataStructure]
            Class of the concrete data structure being created.

        Notes
        -----
        Responsibilities of this method:

        - Ensure that the required class-level attributes are defined in each data structure
          subclass.
        - Declare lazily-initialized attributes (data and coordinates) by assigning the descriptor
          `LazyAttribute` (*after* having checked the presence of the class-level attribute `coords`
          among the required attributes).
        """
        # Call parent hook (default behavior)
        super().__init_subclass__()
        # Check class-level attributes
        for class_attr in cls._REQUIRED_ATTRIBUTES:
            if not hasattr(cls, class_attr):
                raise TypeError(f"<{cls.__name__}> Missing class-level attribute: '{class_attr}'.")
        # Declare lazily-initialized attributes
        setattr(cls, "data", LazyAttribute("data"))
        for coord_name in cls.coords:
            setattr(cls, coord_name, LazyAttribute(coord_name))

    # --- Instance-Level Manipulations -------------------------------------------------------------

    def __init__(
        self,
        data: Optional[Union[CoreData, np.ndarray]] = None,
        **coords_args: Optional[Union[Coordinate, np.ndarray]],
    ) -> None:
        """
        Instantiate a data structure and check the consistency of the input values (automatic).

        Parameters
        ----------
        data : Optional[CoreData, np.ndarray]
            Core data values to fill the attribute `data`.
            Shape: Consistent with the dimensions expected for the data structure.
        coords_args : Dict[str, Optional[Coordinate, np.ndarray]]
            Coordinate values specific to the data structure subclass.
            Keys: Coordinate names as specified in `coords_to_dims`.
            Values: Coordinate values, corresponding to the expected type of coordinate.
            .. _coord_args:

        Implementation
        --------------
        To pass each individual coordinate to the setter method `set_coords`, it is necessary to use
        the unpacking operator `**` and a dictionary rather than the following syntax:

        ``self.set_coords(coord_name=coords_args[coord_name])``

        Indeed, ``coord_name`` would always be treated as the *literal string* ``"coord_name"``
        instead of the actual *value* of the variable ``coord_name``.
        """
        # Lazy initialization: declare private and empty content-related attributes
        self._data: Optional[CoreData] = None
        for coord_name in self.coords:
            setattr(self, f"_{coord_name}", None)
        # Fill with actual values if provided
        if data is not None:
            self.set_data(data)
        for coord_name, coord_value in coords_args.items():
            if coord_value is not None:  # pass individual coordinates to the setter method
                self.set_coords(**{coord_name: coord_value})

    def __repr__(self) -> str:
        # NOTE: Use private attributes to examine the state of the data structure
        data_status = "empty" if self._data is None else "filled"
        active_coords = ", ".join(
            [name for name in self.coords if getattr(self, f"_{name}") is not None]
        )
        return f"<{self.__class__.__name__}> Dims: {self.dims}, Data: {data_status}, Active coords: {active_coords}"

    # --- Access to Attributes ---------------------------------------------------------------------

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

        Parameters
        ----------
        data : Union[CoreData, np.ndarray]
            See the attribute `data`.

        Raises
        ------
        ValueError
            If the number of dimensions of the input data is not consistent with the dimensions
            expected for the data structure.
        """
        if data.ndim != len(self.dims):
            raise ValueError(f"Invalid number of dimensions: data {data.ndim} != {len(self.dims)}")
        if not isinstance(data, CoreData):  # convert to CoreData with the expected dims
            data = CoreData(data, self.dims)
        self._data = data  # store in private attribute (bypass descriptor)

    def set_coords(self, **coords_args: Union[Coordinate, np.ndarray]) -> None:
        """
        Set coordinate attributes (all or a subset) and check their consistency.

        Parameters
        ----------
        coords_args : Dict[str, Union[Coordinate, np.ndarray]]
            See the argument :ref:`coords_args` in the constructor.

        Raises
        ------
        ValueError
            If any argument has an unexpected coordinate name.
        ValueError
            If the shape of the value passed for one coordinate does not match the sub-shape of the
            axis to which it is associated in tha data structure.

        See Also
        --------
        :meth:`CoreData.get_size`: Get the length of a dimension.
        """
        for name, value in coords_args.items():
            if name not in self.coords:
                raise ValueError(f"Invalid coordinate name: '{name}'.")
            if not isinstance(value, self.coords[name]):  # convert to the expected coordinate type
                value = self.coords[name](value)
            if self._data is not None:  # check consistency with the data structure dimensions
                valid_shape = tuple(self.get_size(dim) for dim in self.coords_to_dims[name])
                if value.shape != valid_shape:
                    raise ValueError(f"Invalid shape for '{name}': {value.shape} != {valid_shape}")
            setattr(self, f"_{name}", value)  # store in private attribute (bypass descriptor)

    # --- Data Manipulations -----------------------------------------------------------------------

    def copy(self) -> Self:
        """Return a *deep copy* of the data structure."""
        return copy.deepcopy(self)

    def sel(self, **kwargs) -> Self:
        """
        Select data along specific coordinates.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Keys: Coordinate names.
            Values: Selection criteria (single value, list or slice).

        Returns
        -------
        DataStructure
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
        # raise NotImplementedError("Not implemented yet.")
        return self

    # --- I/O Handling -----------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """
        PLACEHOLDER METHOD - Path to the file containing the data.

        Warning
        -------
        This property is abstract as soon as the class-level attribute `path_loader` is set.
        Implement it by passing the required arguments to the path ruler.
        """
        return self.path_ruler().get_path()

    def load(self) -> None:
        """Load an instance from the file at :attr:`path`."""
        loaded_obj = self.loader(path=self.path, tpe=self.tpe).load()
        self.__dict__.update(loaded_obj.__dict__)

    def save(self) -> None:
        """Save an instance to a file at :attr:`path` in the format specific to the saver."""
        self.saver(self.path, self).save()
