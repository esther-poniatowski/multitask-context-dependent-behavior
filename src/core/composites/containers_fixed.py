#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.containers_fixed` [module]

Classes
-------
UnitsContainer
ExpCondContainer
"""
from typing import List, Iterable, Type

from core.composites.base_container import Container, V
from core.composites.exp_conditions import ExpCondition
from core.attributes.brain_info import Unit


class UnitsContainer(Container[Unit, V]):
    """
    Container for population data.

    Class Attributes
    ----------------
    KEY_TYPE : Type[Unit]
        Type of the keys in the container, here `Unit`.

    Examples
    --------
    Initialize a `UnitsContainer` with a dictionary of integer values:

    >>> units_to_values = {Unit("avo052a-d1"): 1, Unit("lemon052a-b2"): 2}
    >>> container = UnitsContainer(units_to_values, value_type=int)

    Initialize a `UnitsContainer` with two lists of units and values:

    >>> units = [Unit("avo052a-d1"), Unit("lemon052a-b2")]
    >>> values = [1, 2]
    >>> container = UnitsContainer(units, values, value_type=int)

    """

    KEY_TYPE = Unit

    def __init__(self, *args, value_type: Type[V] | None = None, **kwargs) -> None:
        """Override the base constructor to fix `key_type` and allow dynamic `value_type`."""
        super().__init__(*args, key_type=self.KEY_TYPE, value_type=value_type, **kwargs)

    @classmethod
    def from_keys(
        cls, keys: Iterable[Unit], fill_value: V, *, value_type: Type[V] | None = None, **kwargs
    ) -> "UnitsContainer[V]":
        """Override the base class method to fix `key_type` and allow dynamic `value_type`."""
        return cls({key: fill_value for key in keys}, value_type=value_type, **kwargs)

    @property
    def units(self) -> List[Unit]:
        """Get the units in the container."""
        return list(self.data.keys())


class ExpCondContainer(Container[ExpCondition, V]):
    """
    Container for experimental conditions.

    Class Attributes
    ----------------
    KEY_TYPE : Type[ExpCondition]
        Type of the keys in the container, here `ExpCondition`.

    Examples
    --------
    Initialize an `ExpCondContainer` with a dictionary of integer values:

    >>> exp_conds_to_values = {ExpCondition("a"): 1, ExpCondition("b"): 2}
    >>> container = ExpCondContainer(exp_conds_to_values, value_type=int)

    """

    KEY_TYPE = ExpCondition

    def __init__(self, *args, value_type: Type[V] | None = None, **kwargs) -> None:
        """Override the base constructor to fix `key_type` and allow dynamic `value_type`."""
        super().__init__(*args, key_type=self.KEY_TYPE, value_type=value_type, **kwargs)

    @classmethod
    def from_keys(
        cls,
        keys: Iterable[ExpCondition],
        fill_value: V,
        *,
        value_type: Type[V] | None = None,
        **kwargs,
    ) -> "ExpCondContainer[V]":
        """Override the base class method to fix `key_type` and allow dynamic `value_type`."""
        return cls({key: fill_value for key in keys}, value_type=value_type, **kwargs)

    @property
    def exp_conditions(self) -> List[ExpCondition]:
        """Get the experimental conditions in the container."""
        return list(self.data.keys())
