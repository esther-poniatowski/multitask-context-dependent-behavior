#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io.savers` [subpackage]

Classes to save data to files.

Modules
-------
:mod:`base`
:mod:`impl`

Examples
--------
Save DataFrame to CSV

.. code-block:: python
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    saver = SaverCSV(path='path/to/data', data)
    saver.save()
"""
