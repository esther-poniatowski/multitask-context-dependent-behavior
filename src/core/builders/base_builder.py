#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.base_builder` [module]

Base classes for data structure builders.

Classes
-------
`DataStructureBuilder`

Examples
--------
Implementation of a concrete builder class:

.. code-block:: python

    class ConcreteBuilder(DataStructureBuilder):
        data_class = ConcreteDataStructure

        def build(self,
                  input_for_data: np.ndarray,
                  input_for_coord: np.ndarray,
                  input_metadata: str
        ) -> ConcreteDataStructure:
            data_pipeline = DataPipeline()
            coord_pipeline = CoordPipeline()
            data = data_pipeline.process(input_for_data)
            coord = coord_pipeline.process(input_for_coord)
            return self.data_class(data=data, coord=coord, metadata=input_metadata)

Usage of the concrete builder:

.. code-block:: python

        builder = ConcreteBuilder()
        data_structure = builder.build(input_for_data, input_for_coord, input_metadata)

"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Tuple, Optional

import numpy as np

from core.coordinates.base_coord import Coordinate
from core.data_structures.core_data import CoreData, DimName
from core.data_structures.base_data_struct import DataStructure


Product = TypeVar("Product", bound=DataStructure | CoreData | Coordinate)
"""Type variable for the Data structure class produced by a specific builder."""


class Builder(Generic[Product], ABC):
    """
    Abstract base class for building high-level objects: data structures, core data, coordinates.

    Class Attributes
    ----------------
    PRODUCT_CLASS : type
        Class of the product to build.
    TMP_DATA : Tuple[str]
        Names of the internal data attributes used by the builder (if any, for temporary storage).

    Attributes
    ----------
    product : Optional[Product]
        Product instance being built.

    Methods
    -------
    `build`
    `reset`
    `get_product`

    Implementation
    --------------
    Separation of concerns between the builder's methods:

    - Constructor: Declare the type of product to build (base class' constructor) and store
      static configuration parameters (subclasses' constructors) which determine the behavior of the
      builder across multiple builds.
    - Main `build()` method: Receive dynamic inputs required to build a specific product instance.
      Metadata (which only serve to determine the identity of the data structure product) are
      transferred to the constructor of the data structure itself. Inputs to process are stored as
      attributes in the builder for direct access during the building process. This method
      orchestrates the creation of a data structure step-by-step and returns the complete product
      instance.
    - Specific methods to process internal data: Only deal with the data they receive from
      `build()`.
    """

    PRODUCT_CLASS: Type[Product]
    TMP_DATA: Tuple[str, ...]

    def __init__(self) -> None:
        self.product: Optional[Product] = None  # declare the type of product to build
        self.reset()

    def reset(self) -> None:
        """Reset the builder's state: clear the product and internal data."""
        self.product = None
        for attr in self.TMP_DATA:
            setattr(self, attr, None)

    def get_product(self) -> Product:
        """
        Return the product after complete building and reset the builder's state.

        Returns
        -------
        product
            Product instance built by this builder.
        """
        assert self.product is not None
        product = self.product
        self.reset()
        return product

    @abstractmethod
    def build(self, **kwargs) -> Product:
        """
        Orchestrate the creation of a product step-by-step.

        Arguments
        ---------
        args, kwargs:
            Specific input objects required to build the product (data, metadata...).

        Returns
        -------
        product
            Product instance built by this builder.

        Notes
        -----
        This method has the responsibility of an equivalent Director class in the classical Builder
        design pattern, which defines a sequence of construction steps to create specific kinds of
        products.
        Usually, the product data structure is initialized at the beginning of the method for
        incremental construction. Each step can immediately update the product as soon as a
        component becomes available, without requiring temporary storage.
        """


ProductCoordinate = TypeVar("ProductCoordinate", bound=Coordinate)
"""Type variable for the Coordinate class produced by a specific builder."""


class CoordinateBuilder(Builder[ProductCoordinate]):
    """
    Abstract base class for building coordinates.

    Class Attributes
    ----------------
    Same as the `Builder` class.

    Attributes
    ----------
    Same as the `Builder` class.

    Methods
    -------

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`: Abstract base class for coordinates.
    """

    # No specific method yet


ProductCoreData = TypeVar("ProductCoreData", bound=CoreData)
"""Type variable for the Core Data class produced by a specific builder."""


class CoreDataStructureBuilder(Builder[ProductCoreData]):
    """
    Abstract base class for building core data arrays.

    Class Attributes
    ----------------
    Same as the `Builder` class.

    Attributes
    ----------
    Same as the `Builder` class.

    Methods
    -------

    See Also
    --------
    `core.data_structures.core_data.CoreData`: Class for core data arrays.
    """

    # No specific method yet


ProductDataStructure = TypeVar("ProductDataStructure", bound=DataStructure)
"""Type variable for the Data structure class produced by a specific builder."""


class DataStructureBuilder(Builder[ProductDataStructure]):
    """
    Abstract base class for building data structures.

    Class Attributes
    ----------------
    Same as the `Builder` class.

    Attributes
    ----------
    Same as the `Builder` class.

    Methods
    -------
    `get_dimensions`
    `initialize_data_structure`
    `initialize_core_data`
    `add_data`
    `add_coords`

    See Also
    --------
    `core.data_structures.base_data_struct.Data`: Abstract base class for data structures.

    Implementation
    --------------
    The builder fills data and coordinates using the methods of the data structure interface:
    `set_data` and `set_coords`. Those methods perform automatic validations so that the builder can
    focus on the creation process.
    """

    def get_dimensions(self) -> Tuple[DimName, ...]:
        """
        Retrieve the dimensions of the data structure product by delegating to the product class.

        Required to initialize the CoreData.
        """
        return self.PRODUCT_CLASS.dims

    @abstractmethod
    def initialize_data_structure(self, **metadata) -> None:
        """
        Initialize an empty data structure product *with its metadata*.

        Arguments
        ---------
        metadata : dict
            Metadata corresponding to the arguments *required* by the constructor of the data
            structure class. It includes:

            - Identifiers which uniquely characterize the data structure product.
            - Descriptive parameters about configuration or content (if any).
        """
        self.product = self.PRODUCT_CLASS(**metadata)

    def initialize_core_data(self, shape: Tuple[int, ...]) -> CoreData:
        """
        Initialize the core data array with empty values and the dimensions names inherent to the
        class of the data structure product.

        Returns
        -------
        data : CoreData

        See Also
        --------
        :func:`numpy.full`

        Warning
        -------
        Here, `np.full` is used instead of `np.empty` to ensure that the content of the array is
        predictable. With `np.empty`, the array is filled with random values that could remain if
        filling is only partial.
        Initialization with ``NaN`` values is only suitable for floating types. For integer types,
        override this method. To mark uninitialized elements, use either a sentinel value state or
        masked arrays.
        """
        data = CoreData(np.full(shape, np.nan), dims=self.get_dimensions())
        return data

    def add_data(self, data: CoreData) -> None:
        """
        Add actual data values to the data structure product via the data structure's setter.

        Arguments
        ---------
        data: CoreData
            Data values to store in the data structure's attribute `data`.

        See Also
        --------
        `core.data_structures.core_data.CoreData`
        `core.data_structures.base_data_struct.DataStructure.set_data`
        """
        assert self.product is not None
        self.product.set_data(data=data)

    def add_coords(self, **coords: Coordinate) -> None:
        """
        Add actual coordinates to the data structure product via the data structure's setter.

        Arguments
        ---------
        coords: Coordinate
            Coordinate objects to store in the data structure's attributes corresponding to the keys
            in the dictionary.

        See Also
        --------
        `core.coordinates.base_coord.Coordinate`
        `core.data_structures.base_data_struct.DataStructure.set_coords`
        """
        assert self.product is not None
        self.product.set_coords(**coords)
