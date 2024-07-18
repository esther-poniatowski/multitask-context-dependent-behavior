#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates` [subpackage]

Classes representing coordinates (labels) associated to the data structures.

**Content and Functionalities of Coordinates**

Coordinates are designed to represent

Each class of coordinate represents one family of coordinate, which corresponds to a common type of
labels associated with the dimensions of the data sets. (e.g. time stamps, tasks, context,
stimuli...). Each instance of a coordinate class represents one specific set of labels within the
family (e.g. set of stimuli on each trial in an experiment).

Each coordinate object encapsulates :

- Labels associated with one dimension of the data set
- General metadata describing the constraints inherent to the coordinate family
- Specific metadata describing the unique properties of the coordinate instance
- Methods relevant to manipulate the coordinate labels

**Creating Coordinate**

A coordinate object can be obtained via two pathways :

+-----------+--------------------------------------+--------------------------------------+
|           | Generating basic labels              | Injecting pre-computed labels        |
+===========+======================================+======================================+
| Approach  | ``Coordinate.create(...)``           | ``Coordinate(values, ...)``          |
|           | with minimal arguments               | with both values and arguments       |
+-----------+--------------------------------------+--------------------------------------+
| Use Cases | In tests, to get ready-to-use        | In processing pipelines, to build    |
|           | coordinate objects                   | complex composite coordinates for    |
|           |                                      | specific data sets                   |
+-----------+--------------------------------------+--------------------------------------+


Modules
-------
:mod:`base`
:mod:`time`
:mod:`exp_condition`
:mod:`exp_structure`
:mod:`trials`
:mod:`bio`


Implementation
--------------
Uniform Interface for Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All coordinates adhere to a *uniform* interface, which is used throughout the package.
This interface is established in the abstract base class :class:`Coordinate.

Implementing Specific Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each specific type of coordinate is implemented as a concrete subclass which inherits from
:class:`Coordinate`.

To be valid, each coordinate subclass must perform the following steps:

Implement a Custom Constructor
...............................

- Admit an argument ``values`` and pass it to the base constructor
- Set additional attributes (if needed) to store metadata along with the coordinate labels.

Comply with the Interface
.........................

- Implement the required abstract methods (e.g. :meth:`build_labels`).


Customize the Coordinate Class (Optional)
.........................................

- Override some base class methods to adapt the behavior (e.g. :meth:`__repr__`).
- Add methods to add manipulations specific to the coordinate family (e.g. :meth:`count_by_lab`).


Notes
-----
Coordinates do not store attributes which would depend on the data structure in which they are
embedded (e.g. name of the associated dimension in the data structure instance). This pairing is
managed by the data structures themselves. This design enhances modularity and decouples the
coordinate classes from the data set classes to which they apply.


Examples
--------
Creating basic coordinates:

>>> CoordTime.create(n_smpl=10, t_max=1)
<CoordTime> : 10 time points at 0.1 s bin.

>>> CoordTask.create(n_smpl=10, cnd=Task.PTD)
<CoordTask> : 10 samples, Task.PTD: 10.

>>> CoordFold.create(n_smpl=10, k=5)
<CoordFold> : 10 samples, [2, 2, 2, 2, 2].

Creating custom coordinates:

>>> values = np.array([True, False, True])
>>> CoordError(values)
<CoordError> : 3 samples, Correct: 1, Error: 2.
"""
