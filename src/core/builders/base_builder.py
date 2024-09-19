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
from typing import TypeVar, Generic, Type

from core.data_structures.base_data_struct import DataStructure


D = TypeVar("D", bound=DataStructure)
"""Type variable representing the Data structure class associated with each builder."""


class DataBuilder(Generic[D], ABC):
    """
    Abstract base class for building data structures.

    Class Attributes
    ----------------
    data_class: type
        Class of the data structure to build.

    Attributes
    ----------
    data_structure: D
        Data structure instance built by this builder.

    Methods
    -------
    :meth:`build`

    See Also
    --------
    :class:`core.data_structures.base_data_struct.Data`
        Abstract base class for data structures.

    Implementation
    --------------
    The inputs are passed to the `build()` method rather than to the constructor, which allows to
    reuse the same builder instance for building different instances of the product with different
    inputs.
    """

    data_class: Type[D]

    def __init__(self) -> None:
        self.data_structure: D

    @abstractmethod
    def build(self, *args, **kwargs) -> D:
        """
        Finalize the creation of the data structure by processing inputs through pipelines.

        Arguments
        ---------
        args, kwargs:
            Specific input objects required to build the product (data, metadata, etc.).

        Returns
        -------
        product_type
            Data structure instance built by this builder.
        """
