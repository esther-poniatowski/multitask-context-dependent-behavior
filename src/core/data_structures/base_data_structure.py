#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.base_data_structure` [module]

Classes
-------
DataStructure (abstract base class)

Implementation
--------------
Constraints for Subclasses:

The `__subclass_init__` method of the data structure base class declares the required class-level
attributes that should be defined in each subclass's body. This is a lightweight alternative to the
metaclass approach.

Separation of concerns between the base and subclasses constructors:

- Base class constructor: Declare and (optionally) set the data component attributes (core data and
  coordinates).
- Subclass constructors: Assign the metadata attributes (identifiers for the data structure and
  optional descriptive information) AND call the base class constructor with the data component
  arguments.
"""
from abc import ABC
import copy
from typing import Tuple, Mapping, Self, FrozenSet, Set, TypeVar, Generic, Generator

from core.data_components.core_dimensions import Dimensions, DimensionsSpec
from core.data_components.base_data_component import DataComponent, ComponentSpec
from core.data_components.core_metadata import MetaDataField
from core.data_components.core_data import CoreData
from core.coordinates.base_coordinate import Coordinate

AnyCoreData = TypeVar("AnyCoreData", bound=CoreData)
"""Type variable for the core data component stored in the data structure."""


class DataStructure(ABC, Generic[AnyCoreData]):
    """
    Abstract base class for data structures, defining the interface to interact with data.

    Class Attributes
    ----------------
    DIMENSIONS_SPEC : DimensionsSpec
        Specification of the dimensions of the data structure class (names, order, required and
        optional).
        To be defined in each subclass.
    COMPONENTS_SPEC : ComponentSpec
        Specification of the data components allowed in the data structure class (names and types).
        To be defined in each subclass.
    IDENTIFIERS : FrozenSet[str]
        Names of the metadata attributes which jointly and uniquely identify each data structure
        instance within its class. Handled by each subclass' constructor.
    REQUIRED_IN_SUBCLASSES : Tuple[str]
        Names of the class-level attributes which have to be defined in each data structure
        subclass.

    Attributes
    ----------
    dims : Dimensions
        Registry of actual dimensions in an instance (subset of the `DIMENSIONS_SPEC` attribute).
        Updated as new data components are added to the data structure (see the
        `register_dimensions` method).
    coords : Set[str]
        Registry of active coordinates in the data structure instance (subset of the
        `COMPONENTS_SPEC` attribute). Updated as new coordinates are added to the data structure
        (see the `register_coord` method).
    data : CoreData
        Actual data values to analyze.
        Shape: As many dimensions as the length of the `dims` attribute (by construction).
    shape : Tuple[int]
        (Property) Shape of the data array (delegated to the `CoreData` object).

    Methods
    -------
    get_data
    get_coord
    get_coords_from_dim
    iter_coords
    __getattr__
    set_data
    set_coord
    register_coord
    register_dimensions
    copy
    sel

    Examples
    --------
    Get the name of an axis (two equivalent approaches):

    >>> data.dims[0]
    "time"
    >>> data.get_dim(0) # delegated to the dimension object
    "time"

    Get the axis of the time dimension:

    >>> data.get_axis('time') # delegated to the dimension object
    0

    Get the size of the time dimension:

    >>> data.get_size('time') # delegated to the core data object
    10

    Get the time coordinate:

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
    Hierarchy of dimensions in data structures and their components:

    - The general dimensions for the data structure class are specified in the `DIMENSIONS_SPEC`
      class attribute. It sets the names of the allowed dimension, their order, and whether they are
      required or optional.
    - Each class instance stores its actual dimensions under the `dims` attribute, which is a
      `Dimensions` object containing a subset of the dimensions in the `DIMENSIONS_SPEC` class-level
      attribute.
    - Each of the data component nested in a data structure instance owns its `Dimensions` object,
      which should be consistent with the instance-level `dims` attribute.

    Setting data components (core data and coordinates):

    - Lazy initialization: Data component attributes can be set at the initialization of the data
      structure, or later via the dedicated setter methods (`set_data` and `set_coord`). This
      behavior allows to construct data structures step by step in builders. If no values are
      provided at initialization, the attributes are *not* set to `None` to avoid type checking
      errors in methods which expect the attributes to be filled.
    - Expected types are specified in and the class-level attribute `COMPONENTS_SPEC`. In the base
      constructor, type annotations are specified for the `data` attribute (declaration, without
      actual initialization if no values are provided).
    - Validation of the input values is performed by the setter methods, which are themselves called
      by the constructor if the arguments are provided.

    Accessing attributes:

    - To access the data components, use the dedicated getter methods: `get_data` and `get_coord`
      (using its attribute name). Those methods raise an error in case the attribute is not set, to
      ensure that the client code operates on valid objects (instead of `None`).
    - To access the attributes or methods of the nested objects (delegation):
       - Use specific methods implemented in the `DataStructure` class: `shape` (property),
         `get_dim`, `get_axis`, `get_size`.
       - Use the dot notation or the `getattr` function to access any method or attribute which is
         not explicitly handled by the data structure class but exists in the nested objects. This
         is possible because the `DataStructure` class overrides the `__getattr__` method, which
         iterates through the nested objects to find the requested attribute.

    See Also
    --------
    `Dimensions`: Base class for dimensions.
    `DimensionsSpec`: Specification of the dimensions of a data structure.
    `DataComponent`: Base class for data components.
    `CoreData`: Base class for core data.
    `Coordinate`: Base class for coordinates.
    `ComponentSpec`: Specification of the data components allowed in a data structure.
    """

    # --- Schema of the Data Structure -------------------------------------------------------------

    # Type hints for class-level attributes
    DIMENSIONS_SPEC: DimensionsSpec
    COMPONENTS_SPEC: ComponentSpec = ComponentSpec(data=CoreData)
    IDENTIFIERS: Mapping[str, MetaDataField]
    # Required class-level attributes in each subclass
    REQUIRED_IN_SUBCLASSES = ("DIMENSIONS_SPEC", "COMPONENTS_SPEC", "IDENTIFIERS")

    def __init_subclass__(cls) -> None:
        """
        Hook method called when any subclass of `DataStructure` is created.
        Ensure that the required class-level attributes are defined in each subclass.

        Parameters
        ----------
        cls : Type[DataStructure]
            Class of the concrete data structure being created.
        """
        super().__init_subclass__()  # call parent hook (default behavior)
        for class_attr in cls.REQUIRED_IN_SUBCLASSES:  # check class-level attributes
            if not hasattr(cls, class_attr):
                raise TypeError(f"<{cls.__name__}> Missing class-level attribute: '{class_attr}'.")

    def __init__(self, data: AnyCoreData | None = None, **coords: Coordinate | None) -> None:
        """
        Instantiate a data structure and check the consistency of the input values (automatic).

        Parameters
        ----------
        data : CoreData | None
            Core data values for the attribute `data`.
            Type: Consistent with the type expected from the `COMPONENTS_SPEC` attribute.
            Shape: Consistent with the dimensions expected from the `DIMENSIONS_SPEC` attribute, and
            the other components if already set.
        coords : Coordinate | None
            Coordinates to store along the core data values.
            Keys: Names expected from the `COMPONENTS_SPEC` attribute.
            Type: Consistent with the type expected from the `COMPONENTS_SPEC` attribute.
            Shape: Consistent with the other components if already set.
        """
        # Declare essential attributes
        self.dims: Dimensions = Dimensions()
        self.coords: Set[str] = set()
        self.data: AnyCoreData
        # Fill with actual values if provided (lazy initialization)
        if data is not None:
            self.set_data(data)
        for name, coord in coords.items():
            if coord is not None:
                self.set_coord(name, coord)

    def __repr__(self) -> str:
        data_status = "empty" if not hasattr(self, "data") else "filled"
        active_coords = ", ".join(self.coords) if self.coords else "none"
        return (
            f"<{self.__class__.__name__}> Dims: {self.dims}, "
            f"Data: {data_status}, Coords: {active_coords}"
        )

    # --- Getter Methods ---------------------------------------------------------------------------

    def has_data(self) -> bool:
        """Check if the data attribute is set."""
        return hasattr(self, "data")

    def get_data(self) -> AnyCoreData:
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
        if not self.has_data():
            raise AttributeError(f"Data not set in {self.__class__.__name__} instance.")
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
            If the coordinate name is not among the active coordinates in the data structure.
        """
        if name not in self.coords:
            raise AttributeError(f"Coordinate '{name}' not active in {self.coords}.")
        return getattr(self, name)

    def get_coords_from_dim(self, dim: str) -> Mapping[str, Coordinate]:
        """
        Get all coordinates associated with one dimension of the data structure.

        Returns
        -------
        coords : Dict[str, Coordinate]
            Coordinates associated with the specified dimension of the data structure.
        """
        return {name: coord for name, coord in self.iter_coords() if dim in coord.dims}

    def iter_coords(self) -> Generator[Tuple[str, Coordinate], None, None]:
        """
        Iterate over the active coordinates in the data structure.

        Yields
        ------
        name : str
            Name of the coordinate.
        coord : Coordinate
            Coordinate object stored in the data structure.
        """
        for name in self.coords:
            yield name, getattr(self, name)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the core data (delegate to the core data object)."""
        return self.data.shape

    def get_dim(self, axis: int) -> str:
        """Delegate to the dimension object."""
        return self.dims.get_dim(axis)

    def get_axis(self, name: str) -> int:
        """Delegate to the dimension object."""
        return self.dims.get_axis(name)

    def get_size(self, name: str) -> int:
        """Delegate to the first component object which owns this dimension."""
        if name not in self.dims:
            raise ValueError(f"Dimension '{name}' not active in {self.dims}.")
        if self.has_data():
            return self.data.get_size(name)
        coords_with_dim = self.get_coords_from_dim(name)
        if coords_with_dim:
            return next(iter(coords_with_dim.values())).get_size(name)
        raise ValueError(f"Dimension '{name}' not found in the data structure.")

    def __getattr__(self, name: str):
        """
        Delegate the access to nested attributes and methods of the component objects.

        Parameters
        ----------
        name : str
            Name of the attribute to get.

        Returns
        -------
        Any
            Value of the attribute in the nested object.

        Raises
        ------
        AttributeError
            If the attribute is not found in the data structure or its nested objects.

        Notes
        -----
        The attributes considered are all the active components, the dimensions and the IDENTIFIERS.
        """
        nested_attr = self.coords | {"data", "dims"} | self.identifiers
        for attr in nested_attr:
            obj = getattr(self, attr, None)
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"Invalid attribute '{name}' for '{self.__class__.__name__}'. "
            f"Not in any nested object: {nested_attr}"
        )

    # --- Setter Methods ---------------------------------------------------------------------------

    def set_data(self, data: AnyCoreData) -> None:
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

        See Also
        --------
        `DimensionsSpec.validate`: Check the consistency of the dimensions.
        """
        # Validate type (COMPONENTS_SPEC)
        self.COMPONENTS_SPEC.validate("data", data)
        # Validate dimensions
        self.DIMENSIONS_SPEC.validate(data.dims)
        # Validate shape consistency with the other components
        self.validate_shape(data)
        # Store the data
        self.data = data
        self.register_dimensions(data.dims)

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
        # Validate name and type (COMPONENTS_SPEC)
        self.COMPONENTS_SPEC.validate(name, coord)
        # Validate dimensions
        self.DIMENSIONS_SPEC.validate(coord.dims)
        # Validate shape consistency with the other components
        self.validate_shape(coord)
        # Store the coordinate
        setattr(self, name, coord)
        self.register_coord(name)
        self.register_dimensions(coord.dims)

    def register_coord(self, name: str) -> None:
        """Register a new active coordinate in the data structure, if not already present."""
        if name not in self.coords:
            self.coords.add(name)

    def register_dimensions(self, dims: Dimensions) -> None:
        """Register new dimensions in the data structure, if not already present."""
        for dim in dims:
            if dim not in self.dims:
                self.dims.add(dim)

    def validate_shape(self, component: DataComponent) -> None:
        """Validate the shape of a component before setting it.

        Criteria: The shape of the new component should be consistent with the shape of the other
        components already set in the data structure, along each dimension they have in common.

        Parameters
        ----------
        component : DataComponent
            New component to set in the data structure.

        Raises
        ------
        ValueError
            If the shape of the new component does not match the shape of the other components along
            any common dimension.
        """
        common_dims = Dimensions.intersection(self.dims, component.dims)
        for dim in common_dims:
            expected_size = self.get_size(dim)
            component_size = component.get_size(dim)
            if component_size != expected_size:
                raise ValueError(
                    f"Invalid shape for component '{component.__class__.__name__}' "
                    f"along dimension '{dim}': {component_size} != {expected_size}"
                )

    # --- Data Manipulations -----------------------------------------------------------------------

    def copy(self) -> Self:
        """Create a *deep copy* of the data structure."""
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
