"""
`core.data_structures.base_factory` [module]

Classes
-------
Factory (abstract base class)
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Tuple

from core.data_components.base_data_component import DataComponent


Products = TypeVar("Products", bound=DataComponent | Tuple[DataComponent, ...])
"""Type variable for the product class(es) created by a specific factory."""


class Factory(Generic[Products], ABC):
    """
    Abstract base class for creating intermediate-level data components: core data, coordinates.

    Class Attributes
    ----------------
    PRODUCTS : Tuple[Type[DataComponent], ...]
        Class(es) of the product to create.

    Attributes
    ----------

    Methods
    -------
    create (abstract)
    """

    PRODUCT_CLASSES: Type[DataComponent] | Tuple[Type[DataComponent], ...]

    @abstractmethod
    def create(self, *args, **kwargs) -> Products:
        """
        Orchestrate the creation of one or several coupled product(s).

        Arguments
        ---------
        args, kwargs:
            Specific inputs required to create the products (data, metadata...).

        Returns
        -------
        products : Products
            Product instance(s) built by this factory.
        """
