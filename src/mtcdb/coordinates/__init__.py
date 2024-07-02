#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates` [subpackage]

Classes representing data coordinates (labels) associated to the data structures.

Modules
-------
:mod:`base_coord`
:mod:`time`
:mod:`exp_cond`
:mod:`exp_struct`
:mod:`trials`
:mod:`bio`

Implementation
--------------
- Abstract Base Class
  Coordinate is an abstract base class which defines the common interface 
  for coordinate objects representing data dimensions. 
- Inheritance
  Concrete subclasses inherit from Coordinate to represent specific types of coordinates.
  Each one :
  - Defines its own constructor, which calls the base class constructor
   and (optionally) sets additional attributes (metadata to store along the coordinate).
  - Implements the abstract methods (e.g. ``build_labels``) and (optionally)
    override the base class methods to provide specific functionalities (e.g. ``__repr__``).
  - Adds methods (optionally) to manipulate the specific coordinate labels.
- Arguments
  The base constructor only requires the *values* of the coordinate labels.
  Coordinates do not store attributes which would depend on a particular data set
  (e.g. coordinate name in the data set or associated dimension).
  This pairing is managed by the data structures themselves.
  This design enhances modularity and decouples the coordinate classes
  from the data set classes to which they apply.
"""
