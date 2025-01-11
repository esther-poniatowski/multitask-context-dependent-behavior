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
from typing import Tuple, Mapping, Type, TypeVar, Generic, Self, Union

import numpy as np

from core.data_structures.core_data import CoreData, Dimensions

from core.coordinates.base_coord import Coordinate


C = TypeVar("C", bound=CoreData | Coordinate)
"""Type variable representing the type of content attributes (CoreData, Coordinate)."""


class LazyAttribute(Generic[C]):
    """
    Descriptor for lazy initialization of data and coordinates in the data structure.

    Attributes
    ----------
    name : str
        Name of the lazy attribute to handle.

    Parameters
    ----------
    name : str
        Descriptor attribute `LazyAttribute.name`.
    instance : DataStructure
        Instance of the data structure which stores the attribute.
    owner : Type[DataStructure]
        Class of the data structure.

    Raises
    ------
    AttributeError
        If the lazy attribute is accessed before being populated with actual values.

    Notes
    -----
    Usage of this descriptor:

    - Declare the lazy attributes as class attributes in the data structure base class.
    - Initialize the lazy attributes as private attributes in the data structure constructor.
    - When the attribute is accessed via its public name, the descriptor raises an error if the
      value is not set yet.
    - When a value is assigned to the attribute via the public name, the descriptor raises an error
      to require the use of the dedicated setter method of the data structure class.
    - To bypass the descriptor and set the actual value, use the private attribute name.
    """

    def __init__(self, name):
        self.name = f"_{name}"  # private attribute name

    def __get__(self, instance: "DataStructure", owner: Type["DataStructure"]) -> C:
        value = getattr(instance, self.name)
        if value is None:
            raise AttributeError(f"Unset attribute: '{self.name}'.")
        return value

    def __set__(self, instance: "DataStructure", value: C) -> None:
        raise AttributeError(
            f"Use the setter method of the class {instance.__class__.__name__} "
            f"to set the attribute '{self.name}'."
        )


T = TypeVar("T")
"""Type variable representing the type of data in the generic DataStructure class, i.e., the data structure subclass."""


class DataStructure(Generic[T], ABC):
    """
    Abstract base class for data structures, defining the interface to interact with data.

    Class Attributes
    ----------------
    REQUIRED_ATTRIBUTES : Tuple[str]
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

    Attributes
    ----------
    data : CoreData
        Actual data values to analyze.
    shape : Tuple[int]
        (Property) Shape of the data array (delegated to the CoreData object).

    Methods
    -------
    `get_coord`
    `__getattr__`
    `set_data`
    `set_coords`
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
    `Coordinate`: Base class for coordinates.
    """

    # --- Class-Level Configurations ---------------------------------------------------------------

    # --- Schema of the Data Structure ---
    REQUIRED_ATTRIBUTES = ("dims", "coords", "coords_to_dims", "identifiers")
    dims: Dimensions
    coords: Mapping[str, type]
    coords_to_dims: Mapping[str, Dimensions]
    identifiers: Tuple[str, ...]

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
        for class_attr in cls.REQUIRED_ATTRIBUTES:
            if not hasattr(cls, class_attr):
                raise TypeError(f"<{cls.__name__}> Missing class-level attribute: '{class_attr}'.")
        # Declare lazily-initialized attributes
        setattr(cls, "data", LazyAttribute("data"))
        for coord_name in cls.coords:
            setattr(cls, coord_name, LazyAttribute(coord_name))

    # --- Instance-Level Manipulations -------------------------------------------------------------

    def __init__(
        self,
        data: CoreData | np.ndarray | None = None,
        **coords_args: Coordinate | np.ndarray | None,
    ) -> None:
        """
        Instantiate a data structure and check the consistency of the input values (automatic).

        Parameters
        ----------
        data : CoreData | np.ndarray | None
            Core data values to fill the attribute `data`.
            Shape: Consistent with the dimensions expected for the data structure.
        coords_args : Dict[str, Coordinate | np.ndarray | None]
            Coordinate values specific to the data structure subclass.
            Keys: Coordinate names as specified in `coords_to_dims`.
            Values: Coordinate values, corresponding to the expected type of coordinate.
            .. _coord_args:

        Implementation
        --------------
        Valid syntax to pass coordinates to the setter method `set_coords`:

        ``self.set_coords(**{coord_name: coord_value})``

        Invalid syntax:

        ``self.set_coords(coord_name=coords_args[coord_name])``

        Explanation: ``coord_name`` would always be treated as the *literal string* ``"coord_name"``
        instead of the actual *value* of the variable ``coord_name``.

        The check for `None` values in the `coords_args` dictionary is necessary to allow subclasses
        to define default values for some coordinates. This way, only non-empty coordinates are set
        in the data structure as attributes.

        """
        # Lazy initialization: declare private and empty content-related attributes
        self._data: CoreData | None = None
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
        return (
            f"<{self.__class__.__name__}> Dims: {self.dims}, "
            f"Data: {data_status}, "
            f"Active coords: {active_coords}"
        )

    # --- Access to Attributes ---------------------------------------------------------------------

    def get_data(self) -> CoreData:
        """Get the actual data values."""
        return self.data

    def get_coord(self, name: str) -> Coordinate:
        """
        Retrieve a coordinate using its attribute name. Useful to iterate over coordinates.

        Parameters
        ----------
        name : str
            Name of the coordinate to retrieve.

        Returns
        -------
        coord : Coordinate
            Coordinate object stored in the data structure.
        """
        if name not in self.coords:
            raise AttributeError(f"Invalid coordinate: '{name}' not in {self.coords.keys()}.")
        return getattr(self, name)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the core data (delegate to the core data object)."""
        return self.data.shape

    def __getattr__(self, name: str):
        """
        Delegate the access to nested attributes and methods of the content objects.

        Parameters
        ----------
        name : str
            Name of the attribute to get.
        """
        # Delegate method calls
        if name == "get_dim":
            return self.dims.get_dim
        elif name == "get_axis":
            return self.dims.get_axis
        elif name == "get_size":
            return self.data.get_size
        else:  # Delegate attribute access to content objects
            for obj in self.__dict__.values():
                if hasattr(obj, name):
                    return getattr(obj, name)
            raise AttributeError(f"Invalid attribute '{name}' for '{self.__class__.__name__}'.")

    # --- Set Data and Coordinates -----------------------------------------------------------------

    def set_data(self, data: CoreData | np.ndarray) -> None:
        """
        Set the attribute `data` with actual values and check its consistency.

        Parameters
        ----------
        data : CoreData | np.ndarray
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

    def set_coords(self, **coords_args: Coordinate | np.ndarray) -> None:
        """
        Set coordinate attributes (all or a subset) and check their consistency.

        Parameters
        ----------
        coords_args : Dict[str, Coordinate | np.ndarray]
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
        `CoreData.get_size`: Get the length of a dimension.
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

        >>> data.sel(categ=['R', 'T'])

        Select error trials only:

        >>> data.sel(error=True)

        Select along multiple coordinates:

        >>> data.sel(time=slice(0, 1), task='PTD', categ=['R', 'T'])
        """
        # raise NotImplementedError("Not implemented yet.")
        return self
