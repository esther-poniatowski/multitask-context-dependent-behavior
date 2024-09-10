#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.builders.base` [module]

Base classes for data structure builders.

Classes
-------
:class:`DataBuilder`
"""
from abc import ABC, abstractmethod


class DataBuilder(ABC):
    """
    Abstract base class for building data structures.

    Attributes
    ----------
    components : dict
        Dictionary to store components of the data structure (data, metadata...).

    Methods
    -------
    :meth:`add_data`
    :meth:`add_metadata`(metadata)`
    :meth:`build`
    """

    def __init__(self):
        self.components = {}

    @abstractmethod
    def add_data(self, data):
        """
        Add data to the builder.

        Parameters
        ----------
        data : Any
            Data to be added to the builder.
        """
        pass

    @abstractmethod
    def add_metadata(self, metadata):
        """
        Add metadata to the builder.

        Parameters
        ----------
        metadata : Any
            Metadata to be added to the builder.
        """
        pass

    @abstractmethod
    def build(self):
        """
        Finalize the creation of the data structure.

        Returns
        -------
        Any
            The data structure built by the builder.
        """
        pass

    def _validate(self):
        """
        Validate that all necessary components are present before building the object.
        """
        if "data" not in self.components:
            raise ValueError("Data is missing.")
        if "metadata" not in self.components:
            raise ValueError("Metadata is missing.")
