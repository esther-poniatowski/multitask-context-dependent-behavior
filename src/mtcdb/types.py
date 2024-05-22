#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`types` [module]
=====================

Define custom types for the whole package.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Union, Sequence


NumpyArray = NDArray[np.float64]
"""Numpy array of float64 elements."""

ArrayLike = Union[Sequence[float], NDArray[np.float64]]
"""Array-like object."""

