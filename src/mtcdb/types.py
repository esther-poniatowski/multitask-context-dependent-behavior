#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.types` [module]
===========================

Define custom types for the whole package.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Union, Sequence, TypeAlias


NumpyArray: TypeAlias = NDArray[np.float64]
"""Numpy array of float64 elements."""

ArrayLike: TypeAlias = Union[Sequence[float], NDArray[np.float64]]
"""Array-like object."""

