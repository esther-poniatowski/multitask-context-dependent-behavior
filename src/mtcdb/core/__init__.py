"""
`core` [package]

Package for Multi-Task Context-Dependent Behavior PhD work (2022-2025).

This package performs data analysis of neuronal recordings in ferret brains, collected in two
context-dependent decision-making tasks.

Notes
-----
To promote modularity, scalability, and maintainability in the analysis workflow, the package is
structured into distinct types of classes. Those are organized in a hierarchical structure.

Two main types of classes are defined based on their role in the analysis workflow:

- Static classes: `Attribute`, `Dimensions`, `DataComponent` (`Coordinate`, `CoreData`),
  `DataStructure`, `Container`.
- Dynamic classes: `Processor`, `Factory`, `Builder`, `Pipeline`.

Static Classes
^^^^^^^^^^^^^^
Static classes form the building blocks of the analysis workflow. They are organized into three
hierarchical categories:

1. **Attributes**: Low-level classes (enum-like) representing the basic data types manipulated in
   the analysis. They provide a central reference and documentation for important aspects of the
   experiment or analysis (e.g., brain areas, experimental conditions, units, folds...).
2. **Data Components**: Intermediate-level classes (numpy-array-like) containing the values to
   analyse or their associated labels. They are further refined between `CoreData` and `Coordinate`.
   Each coordinate is typically associated with an attribute, as it describes one experimental
   variable or analysis label.
3. **Data Structures**: High-level classes (dictionary-like) obtained at the milestones of the
   analysis. They store several data components (core data values, coordinates), and metadata.

Dynamic Classes
^^^^^^^^^^^^^^^
Dynamic classes perform the actual analysis tasks and manipulate the static classes. They are
organized into several hierarchical categories:

1. **Processors**: Low-level classes performing focused operations on basic data types (e.g., NumPy
   arrays).
2. **Factories**: Intermediate-level classes responsible for creating data components. Each
   factory executes multiple processors to generate one or several coupled components.
3. **Builders**: High-level classes responsible for producing complex data structures. Each
   builder executes multiple factories to generate the components required by a specific data
   structure, and stores those components within the data structure object.
4. **Pipelines**: Director classes orchestrating an entire workflow. Each pipeline manages data
   loading, saving, and invokes builders to generate milestone objects for the analysis.

Package Structure
^^^^^^^^^^^^^^^^^
This package is structured into sub-packages, each containing one or several related class(es).

Class hierarchies typically conform to a specific interface defined in a base class. This interface
specified common attributes and methods shared across the concrete implementations within that
hierarchy.

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
`data_components`
`coordinates`
`data_structures`
`composites`
`processors`
`factories`
`builders`
`pipelines`

See Also
--------
`test_core`: Tests for the entire package.
"""
# pylint: disable=unused-variable
# pylint: disable=unused-wildcard-import
from importlib.metadata import version

__version__ = version(__package__)

from core.constants import *
