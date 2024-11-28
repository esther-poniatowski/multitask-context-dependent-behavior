#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core` [package]

Package for Multi-Task Context-Dependent Behavior PhD work (2022-2025).

This package performs data analysis of neuronal recordings in ferret brains, collected in two
context-dependent decision-making tasks.

Notes
-----
This core package contains two types of classes:

- Static classes: Attributes, Coordinates, Data Structures.
- Dynamic classes: Processors, Builders, Pipelines.

Both types are organized in a hierarchical structure to promote modularity, scalability, and
maintainability in the analysis workflow.

Static Classes
^^^^^^^^^^^^^^
Static classes form the building blocks of the analysis workflow. They are organized into three
hierarchical categories:

1. **Attributes**: Low-level classes representing the basic data types manipulated in the analysis.
   They provide a central reference and documentation for important aspects of the experiment.
2. **Coordinates**: Intermediate-level classes representing the dimensions of the data. Each
   coordinate is typically associated with an attribute, as it describes one experimental variable.
3. **Data Structures**: High-level classes representing the milestones of the analysis. They
   usually store core data values (to be analyzed), coordinates, and metadata.

Dynamic Classes
^^^^^^^^^^^^^^^
Dynamic classes perform the actual analysis tasks and manipulate the static classes. They are
organized into three hierarchical levels: Processors, Builders, and Pipelines.

1. **Processors**: Low-level classes performing specific computations or tasks on basic data types
   (e.g., NumPy arrays). Each processor encapsulates one distinct operation, breaking down complex
   tasks into manageable components.
2. **Builders**: Intermediate-level classes responsible for producing complex data structures. Each
   builder executes multiple processors to generate the components required by a specific data
   structure, and stores those components within the data structure object.
3. **Pipelines**: High-level classes orchestrating the entire analysis workflow. Each pipeline
   manages data loading, saving, and invokes builders to generate milestone objects for the
   analysis.


Design Choices
^^^^^^^^^^^^^^
This package is structured into sub-packages, each corresponding to a type of class: Attributes,
Coordinates, Data Structures, Processors, Builders, and Pipelines.

Each sub-package contains a base class that defines the interface, common attributes, and methods
shared across the concrete implementations within that sub-package.

Disabled Warnings
^^^^^^^^^^^^^^^^^
In the modules where a subclass implements an abstract method while specializing the method
signature:

``# pylint: disable=arguments-differ``

Initial problems:

- Number of parameters was x in 'AbstractBaseClass.method' and is now y in overriding
  'Subclass.method' method.
- Variadics removed in overriding 'Subclass.method' method.

Reason for silencing: The abstract methods signatures use variadics ``(*args, **kwarg)``. This allow
subclasses to specialize the method signature without violating the Liskov Substitution Principle
(LSP). Calls to any subclass methods with specific signatures through the base class reference will
be correctly dispatched, as ``(*args, **kwargs)`` can handle arbitrary arguments.

Modules
-------
`constants`

Sub-Packages
------------
`attributes`
`coordinates`
`data_structures`
`processors`
`builders`
`pipelines`

See Also
--------
`test_core`: Tests for the entire package.
"""
# pylint: disable=unused-variable
# pylint: disable=unused-wildcard-import

from core.constants import *
