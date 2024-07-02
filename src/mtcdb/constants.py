#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.constants` [module]

Constants and global variables for the whole package.

They are often used as default parameters in functions and class attributes.

Examples
--------
Access a constant in another module:

.. code-block:: python

        from mtcdb.constants import CONSTANT_NAME
"""

# --- Paths ---

PATH_DATA_ROOT: str = ".../data/"
"""Path to the data directory."""

# --- Time and Durations ---

TBIN: float = 50e-3
"""Time bin (in seconds)."""

TMAX: float = 1.0
"""Duration of the recording period (in seconds)."""

SMPL_RATE: float = 25000
"""Sampling rate of the recordings (in seconds).""" 

SAMPL_RATE_BAGUR: float = 31250
"""Sampling rate (in seconds) specific to the recordings by S.Bagur."""

SMOOTH_WINDOW: float = 100e-3
"""Window size for the smoothing function (in seconds)."""

DPRE = 0.6
"""Duration before stimulus onset (in seconds) in pre-processed time courses."""

DPOST = 0.8
"""Duration after stimulus offset (in seconds) in pre-processed time courses."""

DSTIM = 1.0
"""Duration of the stimulus (in seconds) in pre-processed time courses."""

DCLK = 0.75
"""Duration of the click stimulus (in seconds) in the experiments."""

DWARN = 1.25
"""Duration of the warning stimulus (in seconds) in the experiments (TORC in CLK task)."""

DPRESHOCK = 0.4
"""Duration before the shock onset (in seconds) in the experiments."""

DSHOCK = 0.4
"""Duration of the shock (in seconds) in the experiments."""

# DSTIM_0 = 0.4
#"""Duration shorter for first stimulus (in seconds) in the experiments."""

TON = 0.6
"""Time of stimulus onset (in seconds) in the pre-processed time courses."""

TOFF = 1.35
"""Time of stimulus offset (in seconds) in the pre-processed time courses."""

TSHOCK = 1.75
"""Time of shock onset (in seconds) in the pre-processed time courses."""
