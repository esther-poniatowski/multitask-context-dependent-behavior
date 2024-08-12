#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io.loaders` [subpackage]

Classes to load files from files.

Modules
-------
:mod:`base`
:mod:`impl`

Examples
--------
Load Dictionary from Pickle

.. code-block:: python

    loader = LoaderPKL(path='path/to/data', tpe=dict)
    data = loader.load()
    print(data) # Output: {'A': [1, 2, 3], 'B': [4, 5, 6]}
"""
