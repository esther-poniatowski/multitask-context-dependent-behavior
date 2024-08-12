#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_objects.test_bases` [module]

See Also
--------
:mod:`core.core_objects.base`: Tested module.
"""

from types import MappingProxyType
from typing import FrozenSet, Mapping

import pytest

from core.entities.base import CoreObject


# Test Inputs
# -----------

OPTIONS: FrozenSet[int] = frozenset([1, 2, 3])
"""Valid values for the object."""

FULL_LABELS: Mapping[int, str] = MappingProxyType({1: "One", 2: "Two", 3: "Three"})
"""Full labels for the valid values."""


class TestObject(CoreObject):
    """Concrete class for testing the base object."""

    _options = OPTIONS
    _full_labels = FULL_LABELS


# Test Functions
# --------------


def test_init():
    """
    Test for :meth:`__init__`.

    Expected Outputs
    ----------------
    For valid value: TestObject is created with value 1.

    For invalid value: ValueError
    """
    obj = TestObject(1)
    assert obj.value == 1
    with pytest.raises(ValueError):
        obj = TestObject("invalid_value")


def test_get_options():
    """
    Test for :meth:`get_options`.

    Expected Outputs
    ----------------
    OPTIONS: FrozenSet
    """
    assert TestObject.get_options() == OPTIONS


def test_full_label():
    """
    Test for property :attr:`full_label`.

    Expected Outputs
    ----------------
    "One": str
    """
    obj = TestObject(1)
    assert obj.full_label == "One"


def test_eq():
    """
    Test for :meth:`__eq__`.

    Expected Outputs
    ----------------
    True:
        For two objects with the same value.
    False:
        For two objects with different values.
    """
    assert TestObject(1) == TestObject(1)
    assert TestObject(1) != TestObject(2)


def test_hash():
    """
    Test for :meth:`__hash__`.

    Expected Outputs
    ----------------
    True:
        Check containment.
        Indeed, the hash value in the checking process.
    """
    assert TestObject(1) in [TestObject(1), TestObject(2)]
    assert TestObject(1) not in [TestObject(2), TestObject(3)]
