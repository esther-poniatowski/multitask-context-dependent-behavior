#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_objects.test_concrete` [module]

See Also
--------
:mod:`mtcdb.core_objects.bio`
:mod:`mtcdb.core_objects.exp_structure`
"""

from typing import FrozenSet

import pytest

from core.core_objects.bio import Animal
from core.core_objects.exp_structure import Recording


NAIVE_ANIMALS: FrozenSet = frozenset([Animal("mor"), Animal("tan")])
"""Naive animals expected from the definition in :class:`Animal`."""


def test_get_naive():
    """
    Test for :meth:`Animal.get_naive`.

    Expected Outputs
    ----------------
    NAIVE_ANIMALS: FrozenSet
    """
    assert Animal.get_naive() == NAIVE_ANIMALS


def test_init():
    """
    Test for :attr:`Recording._min`.

    Try to initialize a recording with a value below 1.

    Expected Outputs
    ----------------
    ValueError
    """
    with pytest.raises(ValueError):
        obj = Recording(0)


def test_operations():
    """
    Test for :meth:`Recording.__eq__` and :meth:`Recording.__ne__`.

    Expected Outputs
    ----------------
    True: The two recordings are equal.
    False: The two recordings are not equal.
    """
    obj1 = Recording(1)
    obj2 = Recording(1)
    assert obj1 == obj2
    obj2 = Recording(2)
    assert obj1 != obj2
