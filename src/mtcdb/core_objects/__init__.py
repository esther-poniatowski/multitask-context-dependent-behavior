#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.core_obj` [subpackage]

Define classes representing the core objects manipulated in the package.

Each object is associated to a concrete feature describing the experiment.
It constitutes a type in itself, with its own attributes, methods and properties.

Modules
-------
:mod:`exp_cond`
:mod:`exp_struct`
:mod:`bio`
:mod:`composites`

Notes
-----
Distinct object types are represented by either Enum classes or regular classes.
Enum classes are customized to provide additional information about the objects.
Here, an enum class may define :

- Attributes, to specify the authorized values of an instance.
- Class methods (optionally), to retrieve specific subsets of the instances.
- Properties (optionally), to retrieve additional information about the instances.

Examples
--------
Get the labels authorized for Context objects :
>>> print(list(Context._value2member_map_.values()))
Output : ['a', 'p', 'p-pre', 'p-post']
Get the contexts for naive animals : use the class method.
>>> print(Context.naive())
Output : ['p']
Get the full label for the context 'a' : use the property.
>>> print(Context.A.full_label)
Output : 'Active'

See Also
--------
- `Enum <https://docs.python.org/3/library/enum.html>`_
"""
