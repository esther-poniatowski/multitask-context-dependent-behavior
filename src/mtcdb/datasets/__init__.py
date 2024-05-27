#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.datasets` [subpackage]

Define classes representing data structures used in the whole package.

Each data structure encapsulates the relevant data and the methods to manipulate it.
It implements the constraints and properties inherent to those data structures.
When necessary, a data structure owns methods to load, save, and preprocess it.

Each custom data structure stands as a type in itself which can be used 
in docstrings and type hints of other objects.
It is documented here once for all, as a central reference.

Modules
-------
activity:
    Data structures for the neuronal activity (spiking times, firing rates).
"""


