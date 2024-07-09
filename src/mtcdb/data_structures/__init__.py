#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.data_structures` [subpackage]

Classes representing data structures used in the analysis.

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
**Abstract Base Classes and Concrete Implementations**

:class:`Data` is an abstract base class which defines the interface used 
uniformly across the whole package to interact with data.
Each subclass implements one version for a specific data structure.
It stores : 

- Actual data to analyze
- Coordinates representing dimension labels
- Metadata

**Strategy Design Pattern**

Data structures delegate certain operations to external classes : 

- Path generation
  Each data class is associated to one subclass of :class:`PathManager`.
- Loading and Saving data from and to files 
  Each data class can be associated to a :class:Loader and/or Saver subclass
  depending on the file format.

To interact with those external classes, data classes store instances 
of the appropriate objects among their attributes.

**Interactions with Coordinates - Dependency Injection**

Coordinate objects have to be instantiated outside of the data constructors 
and passed as arguments.
This constraint reduces coupling between Data and Coordinate classes.

**Data Instantiation**

A data structure can be created via two pathways :

- Loading data from a file
  Method: ``load``
  Use cases:
      - To recover data at the start of a processing pipeline
      - To visualize data contents

- Creating data from scratch
  Method: `Data(...)` 
  (direct call to the constructor with appropriate arguments)
  Use cases:
      - To create the final product at the end of a processing pipeline
      - To test functions with custom data

See Also
--------
:mod:`mtcdb.coordinates`
    Coordinate classes representing the dimensions of the data structures.
:mod:`mtcdb.io_handlers.path_managers` 
    PathManager classes implementing path generation rules for each data class.
:meth:`mtcdb.io_handlers.loaders`
    Loader classes used to load data from files in various formats.
:meth:`mtcdb.io_handlers.saver`
    Saver classes used to save data to files in a format determined by the data type itself.
"""
