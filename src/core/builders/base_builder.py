#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.base_builder` [module]

Base classes for data structure builders.

Classes
-------
Builder
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Tuple, Optional

from core.coordinates.base_coord import Coordinate
from core.data_structures.core_data import CoreData
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
    def build(self, *args, **kwargs) -> Product:
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
