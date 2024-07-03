#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.datasets` [subpackage]

Classes representing the data structures used in the analysis.

Each custom data structure stands as a type in itself.
It encapsulates the data, constraints and properties inherent to the specific data,
and the relevant methods to manipulate it.
Each data type can be used in docstrings and type hints of other objects.
It is documented here once for all, as a central reference.

Modules
-------
:mod:`mtcdb.data_structures.base`

Implementation
--------------
- Abstract Base Classes and Concrete Implementations
  Data is an abstract base class which defines the interfaces used 
  uniformly across the whole package to interact with data.
  Each subclass of Data implements one version for a specific data structure.
  It stores : 
  - Actual data to analyze
  - Coordinates representing dimension labels (Coordinate subclasses)
  - Metadata
- Strategy Pattern
  Data structures delegate certain operations to external classes : 
  - Path generation (PathManager subclasses)
  - Data loading and saving (Loader and Saver subclasses)
  To do so, they store instances of appropriate objects as attributes.
- Interactions with Coordinates - Dependency Injection
  Coordinate objects have to be instantiated outside of the data constructors 
  and passed as as arguments.
  This constraint reduces coupling between the Data and Coordinate classes.
- Data Instantiation
  A data structure can be created via two pathways :
  - Loading data from a file
    Method: `load()`
    Example use cases:
        - To recover data at the start of a processing pipeline
        - To visualize data contents
  - Creating data from scratch
    Method: `Data(...)` 
    (direct call to the constructor with appropriate arguments)
    Example use cases:
        - To create the final product at the end of a processing pipeline
        - To test functions with custom data

See Also
--------
:mod:`mtcdb.core_objects.coordinates`
    Coordinate classes representing the dimensions of the data structures.
:mod:`mtcdb.io_handlers.path_managers` 
    PathManager classes implementing path generation rules for each data class.
:meth:`mtcdb.io_handlers.loaders`
    Loader classes used to load data from files in various formats.
:meth:`mtcdb.io_handlers.saver`
    Saver classes used to save data to files in a format determined by the data type itself.
"""
