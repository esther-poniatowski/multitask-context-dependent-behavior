#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb_shared.io.path_managers` [subpackage]

Classes to define file path rules and generate paths.

Modules
-------
:mod:`base`
:mod:`impl`

Examples
--------
Define a File Path

.. code-block:: python

    path_root = '/path/to/data/directory'
    unit = 'avo052a-d1'
    session = 'avo052a04_p_PTD'
    pm = RawSpkTimesPath(path_root)
    path = pm.get_path(unit, session)
    print(path)
    # Output: /path/to/data/directory/raw/avo052a-d1/avo052a04_p_PTD
"""
