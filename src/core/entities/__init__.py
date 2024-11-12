#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.entities` [subpackage]

Define classes representing the entities manipulated in the package.

Each object is associated to a concrete feature describing the experiment.
It constitutes a type in itself, with its own attributes, methods and properties.

Modules
-------
:mod:`base_entity`
:mod:`exp_condition`
:mod:`exp_structure`
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
Get the valid labels for Attention objects :

.. code-block:: python

    from core.entities import Attention
    print(list(Attention.get_options()))
    # Output: ('a', 'p', 'p-pre', 'p-post')

Get the contexts for naive animals :

.. code-block:: python

    print(Attention.naive())
    # Output: ('p')

Get the full label for the attentional state 'a' :

.. code-block:: python

        attn = Attention('a')
        print(attn.full_label)
        # Output: 'Active'
"""
