#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.builders.base_builder` [module]

Base classes for data structure builders.

Classes
-------
:class:`DataBuilder`

Examples
--------
Implementation of a concrete builder class:

.. code-block:: python

    class ConcreteBuilder(DataBuilder):
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
from dataclasses import dataclass
from typing import TypeVar, Generic, Type, Tuple, Optional

# from core.coordinates.base_coord import Coordinate
from core.data_structures.base_data_struct import Coordinate
from core.data_structures.core_data import CoreData, DimName
from core.data_structures.base_data_struct import DataStructure


@dataclass
class DataBuilderInput:
    """
    Base class for builder inputs.

    Notes
    -----
    Each subclass of `DataBuilder` should define a dataclass that inherits from this class.
    """


I = TypeVar("I")
"""Type variable for the input data class associated with a specific builder."""

O = TypeVar("O", bound=DataStructure)
"""Type variable for the Data structure class produced by a specific builder."""


class DataBuilder(Generic[I, O], ABC):
    """
    Abstract base class for building data structures.

    Class Attributes
    ----------------
    product_class : type
        Class of the data structure to build.
    TMP_DATA : Tuple[str]
        Names of the internal data attributes used by the builder.

    Attributes
    ----------
    product : O
        Data structure instance being built.

    Methods
    -------
    `build`
    `reset`
    `initialize_data_structure`
    `get_dimensions`
    `add_data`
    `add_coords`
    `get_product`

    See Also
    --------
    :class:`core.data_structures.base_data_struct.Data`
        Abstract base class for data structures.

    Implementation
    --------------
    Separation of concerns between the builder's methods
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    - Constructor: Declare the data structure product to build (base class' constructor) and store
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

    Lazy Instantiation
    ^^^^^^^^^^^^^^^^^^
    The builder should fill data and coordinates using the `set_data` and `set_coords` methods
    provided by the data structure itself. Those methods perform automatic validations so that the
    builder can focus on the creation process.
    """

    product_class: Type[O]
    TMP_DATA: Tuple[str, ...]

    def __init__(self) -> None:
        """Declare the data structure product to build."""
        self.product: Optional[O]
        self.reset()

    def reset(self) -> None:
        """Reset the builder's state: clear the product and internal data."""
        self.product = None
        for attr in self.TMP_DATA:
            setattr(self, attr, None)

    @abstractmethod
    def build(self, **kwargs) -> O:
        """
        Orchestrate the creation of a data structure step-by-step.

        Arguments
        ---------
        args, kwargs:
            Specific input objects required to build the product (data, metadata...).

        Returns
        -------
        product_type
            Data structure instance built by this builder.
        """

    def initialize_data_structure(self, **metadata) -> None:
        """
        Initialize an empty data structure product *with its metadata*.

        Arguments
        ---------
        metadata: dict
            Metadata corresponding to the arguments *required* by the constructor of the data
            structure class.

        Notes
        -----
        Metadata includes:

        - Identifiers which uniquely characterize the data structure product.
        - Descriptive parameters about configuration or content (if any).
        """
        self.product = self.product_class(**metadata)

    def get_dimensions(self) -> Tuple[DimName, ...]:
        """
        Retrieve the dimensions of the data structure product by delegating to the product class.

        Required to initialize the CoreData.
        """
        return self.product_class.dims

    def add_data(self, data: CoreData) -> None:
        """
        Add actual data values to the data structure product via the data structure's setter.

        Arguments
        ---------
        data: CoreData
            Data values to store in the data structure's attribute `data`.

        See Also
        --------
        :meth:`core.data_structures.core_data.CoreData`
        :meth:`core.data_structures.base_data_struct.DataStructure.set_data`
            Setter method provided by the data structure interface.
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
        :meth:`core.coordinates.base_coord.BaseCoord`
        :meth:`core.data_structures.base_data_struct.DataStructure.set_coords`
            Setter method provided by the data structure interface.
        """
        assert self.product is not None
        self.product.set_coords(**coords)

    def get_product(self) -> O:
        """
        Return the data structure product built by this builder and reset the builder's state.

        Returns
        -------
        product_type
            Data structure instance built by this builder.
        """
        assert self.product is not None
        product = self.product
        self.reset()
        return product
