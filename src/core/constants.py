#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.constants` [module]

Constants and global variables for the whole package.

They are often used as default parameters in functions and class attributes.

Examples
--------
Access a constant in another module:

>>> from core.constants import CONSTANT_NAME
>>> print(CONSTANT_NAME)

"""

# --- Time and Durations ---

T_BIN: float = 50e-3
"""Time bin (in seconds)."""

T_MAX: float = 1.0
"""Duration of the recording period (in seconds)."""

SMPL_RATE: float = 25000
"""Sampling rate of the recordings (in seconds)."""

SAMPL_RATE_BAGUR: float = 31250
"""Sampling rate (in seconds) specific to the recordings by S.Bagur."""

SMOOTH_WINDOW: float = 100e-3
"""Window size for the smoothing function (in seconds)."""

D_PRE = 0.6
"""Duration before stimulus onset (in seconds) in pre-processed time courses."""

D_POST = 0.8
"""Duration after stimulus offset (in seconds) in pre-processed time courses."""

D_STIM = 1.0 # why not 0.75, if it is set to the minimum duration D_CLK?
"""Duration of the stimulus (in seconds) in pre-processed time courses."""

D_CLK = 0.75
"""Duration of the click stimulus (in seconds) in the experiments."""

D_WARN = 1.25
"""Duration of the warning stimulus (in seconds) in the experiments (TORC in CLK task)."""

D_PRESHOCK = 0.4
"""Duration before the shock onset (in seconds) in the experiments."""

D_SHOCK = 0.4
"""Duration of the shock (in seconds) in the experiments."""

# D_STIM_0 = 0.4
# """Duration shorter for first stimulus (in seconds) in the experiments."""

T_ON = 0.6
"""Time of stimulus onset (in seconds) in the pre-processed time courses."""

T_OFF = 1.35
"""Time of stimulus offset (in seconds) in the pre-processed time courses."""

T_SHOCK = 1.75
"""Time of shock onset (in seconds) in the pre-processed time courses."""


# --- Trials ---

N_TRIALS_MIN = 5
"""Minimum number of required trials for a unit to be included in the analysis, in each experimental
condition of interest and fold."""

BOOTSTRAP_THRES_PERC = 0.3
"""Threshold percentage of the number of trials to consider the bootstrap method."""

N_FOLDS = 3
"""Number of folds for cross-validation."""
