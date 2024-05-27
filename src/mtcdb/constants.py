#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.constants` [module]

Define constants and variables used globally in the whole package.

Those contstants are accessible in all the modules :

.. code-block:: python
    
        from mtcdb.constants import CONSTANT_NAME

They are often used as default parameters in functions and class attributes.
"""

DATA_PATH: str = ".../data/"
"""Path to the data directory."""

TBIN: float = 0.001
"""Time bin (in seconds)."""

TMAX: float = 1.0
"""Duration of the recording period (in seconds)."""

SMPL_RATE: float = 0.001
"""Sampling rate of the recordings (in seconds)."""


