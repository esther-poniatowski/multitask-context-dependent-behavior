#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.base_factory` [module]

Classes
-------
Factory (abstract base class)
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Tuple, Optional

from core.data_components.base_data_component import DataComponent


Product = TypeVar("Product", bound=DataComponent)
"""Type variable for the product class created by a specific factory."""


class Factory(Generic[Product], ABC):
    """
    Abstract base class for creating intermediate-level data components: core data, coordinates.

    Class Attributes
    ----------------
    PRODUCT_CLASS : type
        Class of the product to create.

    Attributes
    ----------

    Methods
    -------
    create (abstract)
    """

    PRODUCT_CLASS: Type[Product]

    @abstractmethod
    def create(self, *args, **kwargs) -> Product:
        """
        Orchestrate the creation of a product.

        Arguments
        ---------
        args, kwargs:
            Specific inputs required to create the product (data, metadata...).

        Returns
        -------
        product
            Product instance built by this builder.
        """
