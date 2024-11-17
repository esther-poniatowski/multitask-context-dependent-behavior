#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.base_data_struct` [module]

Classes
-------
`DataStructure` (abstract base class)

Implementation
--------------
Constraints for Subclasses:

The `__subclass_init__` method of the data structure base class declares the required class-level
attributes that should be defined in each subclass's body. This is a lightweight alternative to the
metaclass approach.

Separation of concerns between the base and subclasses constructors:

- Base class constructor: Declare and (optionally) set the content-related attributes (data and
  coordinates).
- Subclass constructors: Assign the metadata attributes (identifiers for the data structure and
  optional descriptive information) AND call the base class constructor with the content-related
  arguments.
"""
from abc import ABC
import copy
from typing import Tuple, Mapping, Type, Self

from core.data_structures.core_data import CoreData, Dimensions

from core.coordinates.base_coord import Coordinate


class DataStructure(ABC):
    """
    Abstract base class for data structures, defining the interface to interact with data.

    Class Attributes
    ----------------
    REQUIRED_IN_SUBCLASSES : Tuple[str]
        Names of the class-level attributes which have to be defined in each data structure
        subclass.
    dims : Dimensions
        Names of the dimensions, ordered to label the axes of the underlying data.
    coords : Mapping[str, Type[Coordinate]]
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
    `set_coord`
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

    Notes
    -----
    Setting content-related attributes (core data and coordinates):

    - Lazy initialization: Content-related attributes can be set at the initialization of the data
      structure, or later via the dedicated setter methods (`set_data` and `set_coord`). If no
      values are provided at initialization, the attributes are *not* set to `None` to avoid type
      checking errors in methods which expect the attributes to be filled.
    - Expected types are specified by the constructor for the `data` attribute (declaration, without
      actually initialization if no values are provided) and by the class-level attribute `coords`
      for the coordinates.
    - Validation of the input values is performed by the setter methods, which are themselves called
      by the constructor if the arguments are provided.

    Accessing content-related attributes:

        - Getter methods are provided to access the content-related attributes: `get_data` and
      `get_coord`. The `get_coord` method allows to retrieve a coordinate using its attribute name,
      which is useful to iterate over coordinates (whose names are registered in the `coords`
      class-level attribute).
    - Delegation to nested attributes is implemented by two approaches:
       - Specific methods: `shape` property (delegated to the core data object).
       - Overriding the `__getattr__` method: shortcuts to the dimensions and coordinates (e.g.,
         `get_dim`, `get_axis`, `get_size`).
    - If a content-related attribute is not provided, then:
        - Trying to access it via the dot notation (`self.attr`) or the `getattr` function
          (`getattr(self, 'attr')`) will raise an error (`AttributeError` or `KeyError`) since the
          attribute does not exist in the dictionary of the instance.
        - Trying to access it via the dedicated getter method (`self.get_data()` or
          `self.get_coord()`) will raise an `AttributeError` (custom behavior).

    See Also
    --------
    `Coordinate`: Base class for coordinates.
    """

    # --- Schema of the Data Structure ---
    REQUIRED_IN_SUBCLASSES = ("dims", "coords", "coords_to_dims", "identifiers")
    dims: Dimensions
    coords: Mapping[str, Type[Coordinate]]
    coords_to_dims: Mapping[str, Dimensions]
    identifiers: Tuple[str, ...]

    def __init_subclass__(cls) -> None:
        """
        Hook method called when any subclass of `DataStructure` is created.
        Ensure that the required class-level attributes are defined in each subclass.

        Parameters
        ----------
        cls : Type[DataStructure]
            Class of the concrete data structure being created.
        """
        # Call parent hook (default behavior)
        super().__init_subclass__()
        # Check class-level attributes
        for class_attr in cls.REQUIRED_IN_SUBCLASSES:
            if not hasattr(cls, class_attr):
                raise TypeError(f"<{cls.__name__}> Missing class-level attribute: '{class_attr}'.")

    def __init__(self, data: CoreData | None = None, **coords: Coordinate) -> None:
        """
        Instantiate a data structure and check the consistency of the input values (automatic).

        Parameters
        ----------
        data : CoreData | None
            Core data values for the attribute `data`.
            Shape: Consistent with the dimensions expected for the data structure.
        coords : Dict[str, Coordinate]
            Coordinate values specific to the data structure subclass.
            Keys: Coordinate names as specified in `coords_to_dims`.
            Values: Coordinate values, corresponding to the expected type of coordinate.
            .. _coord_args:
        """
        # Lazy initialization: declare `data` as CoreData
        self.data: CoreData
        # Fill with actual values if provided
        if data is not None:
            self.set_data(data)
        for name, coord in coords.items():
            self.set_coord(name, coord)

    def __repr__(self) -> str:
        data_status = "empty" if not hasattr(self, "data") else "filled"
        active_coords = ", ".join([name for name in self.coords if hasattr(self, name)])
        return (
            f"<{self.__class__.__name__}> Dims: {self.dims}, "
            f"Data: {data_status}, Coords: {active_coords}"
        )

    # --- Access to Attributes ---------------------------------------------------------------------

    def get_data(self) -> CoreData:
        """
        Get the actual data values.

        Returns
        -------
        data : CoreData
            Core data values stored in the data structure.

        Raises
        ------
        AttributeError
            If the data attribute is not set.
        """
        if not hasattr(self, "data"):
            raise AttributeError("Data not set.")
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

        Raises
        ------
        AttributeError
            If the coordinate name is not valid.
        """
        if name not in self.coords:
            raise AttributeError(f"Invalid coordinate: '{name}' not in {self.coords.keys()}.")
        return getattr(self, name)

    def get_coords_from_dim(self, dim: str) -> Mapping[str, Coordinate]:
        """
        Get all coordinates associated with one dimension of the data structure.

        Returns
        -------
        coords : Dict[str, Coordinate]
            Coordinates associated with the specified dimension of the data structure.
        """
        return {
            name: getattr(self, name) for name in self.coords if dim in self.coords_to_dims[name]
        }

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the core data (delegate to the core data object)."""
        return self.data.shape

    def get_dim(self, dim: int) -> str:
        """Delegate to the dimension object."""
        return self.dims[dim]

    def get_axis(self, name: str) -> int:
        """Delegate to the core data object."""
        return self.data.get_axis(name)

    def get_size(self, name: str) -> int:
        """Delegate to the core data object."""
        return self.data.get_size(name)

    def __getattr__(self, name: str):
        """
        Delegate the access to nested attributes and methods of the content objects.

        Parameters
        ----------
        name : str
            Name of the attribute to get.
        """
        for obj in self.__dict__.values():
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"Invalid attribute '{name}' for '{self.__class__.__name__}'.")

    # --- Set Data and Coordinates -----------------------------------------------------------------

    def set_data(self, data: CoreData) -> None:
        """
        Set the attribute `data` with actual values after validation.

        Parameters
        ----------
        data : CoreData
            See the attribute `data`.

        Raises
        ------
        TypeError
            If the argument is not a `CoreData` object.
        ValueError
            If the dimensions of the input data are not consistent with the dimensions expected by
            the data structure.
        """
        if not isinstance(data, CoreData):
            raise TypeError(f"Invalid type for in DataStructure: {type(data)} != CoreData")
        if data.dims != len(self.dims):
            raise ValueError(f"Invalid number of dimensions: data {data.ndim} != {len(self.dims)}")
        self.data = data

    def set_coord(self, name: str, coord: Coordinate) -> None:
        """
        Set one coordinate attribute after validation.

        Parameters
        ----------
        name : str
            Name of the attribute for the coordinate to set.
        coord : Coordinate
            Coordinate object to set.

        Raises
        ------
        AttributeError
            If the coordinate name is not valid.
        TypeError
            If the argument is not a `Coordinate` object of the subtype expected for this attribute.
        ValueError
            If the shape of the coordinate object does not match the sub-shape of the axis to which
            it is associated in tha data structure.

        See Also
        --------
        `CoreData.get_size`: Get the length of a dimension.
        """
        if name not in self.coords:
            raise AttributeError(f"Invalid coordinate name: '{name}' not in {self.coords.keys()}.")
        if not isinstance(coord, self.coords[name]):
            raise TypeError(f"Invalid type for '{name}': {type(coord)} != {self.coords[name]}")
        if hasattr(self, "data"):  # check consistency with the data structure dimensions
            valid_shape = tuple(self.data.get_size(dim) for dim in self.coords_to_dims[name])
            if coord.shape != valid_shape:
                raise ValueError(f"Invalid shape for '{name}': {coord.shape} != {valid_shape}")
        setattr(self, name, coord)  # store in private attribute (bypass descriptor)

    # --- Data Manipulations -----------------------------------------------------------------------

    def copy(self) -> Self:
        """Return a *deep copy* of the data structure."""
        return copy.deepcopy(self)

    # TODO: Implement the selection method
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
