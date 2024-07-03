#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates` [subpackage]

Classes representing coordinates (labels) associated to the data structures.

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
**Abstract Base Class and Inheritance**

The abstract base class :class:`Coordinate` defines the common interface 
for coordinate objects representing data dimensions. 
Concrete subclasses represent specific types of coordinates.
Each one :
- Defines its own constructor, which calls the base class constructor
  and (optionally) sets additional attributes (metadata to store along with the coordinate).
- Implements the abstract methods (e.g. :meth:`build_labels`) and (optionally)
  override the base class methods to provide specific functionalities (e.g. :meth:`__repr__`).
- Adds methods (optionally) to manipulate the specific coordinate labels.

**Arguments**

Coordinates do not store attributes which would depend on a particular data set
(e.g. coordinate name in the data set or associated dimension).
This pairing is managed by the data structures themselves.
This design enhances modularity and decouples the coordinate classes
from the data set classes to which they apply.
"""
