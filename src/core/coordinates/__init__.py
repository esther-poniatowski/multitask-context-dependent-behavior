#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.coordinates` [subpackage]

Classes representing coordinates (labels) associated to the data structures.

**Content and Functionalities of Coordinates**

Coordinates are designed to represent

Each class of coordinate represents one family of coordinate, which corresponds to a common type of
labels associated with the dimensions of the data sets. (e.g. time stamps, tasks, attentional state,
stimuli...). Each instance of a coordinate class represents one specific set of labels within the
family (e.g. set of stimuli on each trial in an experiment).

Each coordinate object encapsulates :

- Labels associated with one dimension of the data set
- General metadata describing the constraints inherent to the coordinate family
- Specific metadata describing the unique properties of the coordinate instance
- Methods relevant to manipulate the coordinate labels

**Creating Coordinates**

A coordinate object can be obtained via several pathways :

+----------+----------------------------+----------------------------+-----------------------------+
|          | Creating empty coordinates | Generating basic labels    | Injecting custom labels     |
+==========+============================+============================+=============================+
| Approach | ``Coordinate.empty()``     | ``Coordinate.create()``    | ``Coordinate(values, ...)`` |
|          | with no arguments          | with minimal arguments     | with values and arguments   |
+----------+----------------------------+----------------------------+-----------------------------+
| Use      | In empty data structures,  | In tests, to get           | In processing pipelines, to |
| Cases    | to initialize placeholders | ready-to-use               | build composite coordinates |
|          | of the appropriate type    | coordinate objects         | adapted to specific data    |
+----------+----------------------------+----------------------------+-----------------------------+


Modules
-------
:mod:`base_coord`
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

- Admit an argument ``values`` in first position (after ``self``) and pass it to the base
  constructor.
- Set class-specific attributes (if needed) to store metadata along with the coordinate labels. Any
  additional parameter should be assigned a *default value* in the constructor signature. This is
  necessary because the method :meth:`empty` defined in the *base class* calls the *subclass*
  constructor, but remains agnostic to any subclass-specific extra parameter.

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

Disable the pylint error `arguments-differ` for the :meth:`build_labels` method in each concrete
subclass. Indeed, the method signature is necessarily different from the abstract method signature
since it depends on the specific sub-class features.

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
