#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.data_structures` [subpackage]

Classes representing data structures used in the analysis.

Each data structure stands as a custom type in itself. A data structure object encapsulates the
actual data values, the constraints and properties inherent to this specific data set, and the
relevant methods to manipulate it.

Each data structure provides a central documentation. Data types can be used in docstrings and type
hints of other objects.

Modules
-------
:mod:`mtcdb.data_structures.base`

Implementation
--------------
**Abstract Base Classes and Concrete Implementations**

:class:`Data` (abstract base class) defines the interface used uniformly across the package to
interact with data. Each subclass implements one version for a specific data structure. It stores :

- Actual data to analyze
- Coordinates representing dimension labels
- Metadata

**Strategy Design Pattern**

Data structures delegate certain operations to external classes :

- :class:`PathManager` subclasses for path generation.
- :class:`Loader` and :class:`Saver` subclasses for loading/saving data in specific file formats.

To interact with those external classes, data classes store instances of the appropriate objects
among their attributes.

**Interactions with Coordinates - Dependency Injection**

Coordinate objects have to be instantiated outside of the data constructors and passed as arguments.
This constraint reduces coupling between :class:`Data` and :class:`Coordinate` classes.

**Data Instantiation**

A data structure can be obtained via two pathways :

+-----------+--------------------------------------+--------------------------------------+
|           | Loading data from a file             | Creating data from scratch           |
+===========+======================================+======================================+
| Approach  | ``Data(...).load()``                 | ``Data(...)``                        |
|           | with minimal arguments               | with additional arguments to pass    |
|           | to build the path to the file        | data and coordinate values           |
+-----------+--------------------------------------+--------------------------------------+
| Use Cases | - To recover data at the start       | - To create the final product at     |
|           |   of a processing pipeline           |   the end of a processing pipeline   |
|           | - To visualize data contents         | - To test functions with custom data |
+-----------+--------------------------------------+--------------------------------------+


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
