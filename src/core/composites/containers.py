#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.containers` [module]

Classes
-------
`Container`
`UnitsContainer`
`ExpCondContainer`
"""
from collections import UserDict
from typing import List, Callable, Any, Iterable, Tuple, Dict, Self, TypeVar, Generic, Type

from core.entities.bio_info import Unit
from core.composites.exp_conditions import ExpCondition


K = TypeVar("K")
"""Type variable for the keys in the container."""

V = TypeVar("V")
"""Type variable for the values in the container."""

Q = TypeVar("Q")
"""Type variable for the keys in an input dictionary."""

R = TypeVar("R")
"""Type variable for the return type of a function or method applied to the values."""


class Container(UserDict[K, V], Generic[K, V]):
    """
    Container for data, behaving like a dictionary with additional methods.

    Arguments
    ---------
    key_type : Type[K]
        Type of the keys in the container.
    value_type : Type[V]
        Type of the values in the container.
    *args
        Arguments to initialize the dictionary, inherited from `UserDict`.
    **kwargs
        Keyword arguments to initialize the dictionary, inherited from `UserDict`.

    Attributes
    ----------
    key_type : Type[K]
        Type of the keys in the container.
    value_type : Type[V]
        Type of the values in the container.
    data : Dict[K, V]
        Dictionary mapping keys to values, inherited from `UserDict`.

    Raises
    ------
    TypeError
        If the key or value type of an item is not the expected type.

    Methods
    -------
    `__setitem__`
    `list_keys`
    `list_values`
    `get_subset`
    `filter_on_keys`
    `filter_on_values`
    `apply`
    `__getattr__`
    `find_types`

    Examples
    --------
    Initialize a `Container` with a dictionary of data:

    >>> container = Container({1: "a", 2: "b"}, key_type=int, value_type=str)

    Initialize a `Container` with lists of keys and values:

    >>> container = Container([1, 2], ["a", "b"], key_type=int, value_type=str)
    Initialize a `Container` with a dictionary of data and check the types:

    Get a value from the container:

    >>> container[1]
    'a'

    Set a value in the container:

    >>> container[1] = 1
    Traceback (most recent call last):
    ...
    TypeError: Invalid value type: int instead of str

    """

    def __init__(
        self, *args, key_type: Type[K] | None = None, value_type: Type[V] | None = None, **kwargs
    ) -> None:
        """
        Initialize the container with the key and value types.

        Implementation
        --------------
        The parameters `value_type` and `key_type`are keyword-only arguments. Justification for this
        design choice:

        - Reducing ambiguity: It enforces a clearer interface by requiring `key_type` and
          `value_type` to be explicitly named (thereby preventing passing them positionally).
        - Initialization of data: It allows to forward data for the `UserDict` parent class via the
          `*args` argument  before the keyword-only arguments, while `key_type` and `value_type` are
          auxiliary metadata.
        - Flexible Subclassing: Subclasses can override the constructor and fix `key_type` without
          breaking the consistency with the signature and the behavior of the base `Container`
          class.
        """
        if key_type is None:
            raise ValueError("Missing argument: `key_type`")
        if value_type is None:
            raise ValueError("Missing argument: `value_type`")
        self.key_type: Type[K] = key_type
        self.value_type: Type[V] = value_type
        super().__init__(*args, **kwargs)  # pass remaining args and kwargs to UserDict

    def __setitem__(self, key: K, value: V) -> None:
        """
        Set an item in the container, checking the types of the key and value.
        """
        if not isinstance(key, self.key_type):
            raise TypeError(
                f"Invalid key type: {type(key).__name__} instead of {self.key_type.__name__}"
            )
        if not isinstance(value, self.value_type):
            raise TypeError(
                f"Invalid value type: {type(value).__name__} instead of {self.value_type.__name__}"
            )
        super().__setitem__(key, value)

    def list_keys(self) -> List[K]:
        """
        Get a list of keys in the container.

        Returns
        -------
        List[K]
            List of keys in the container.
        """
        return list(self.data.keys())

    def list_values(self, keys: Iterable[K] | None = None) -> List[V]:
        """
        Get a list of values for all or a subset of keys in a specific order.

        Arguments
        ---------
        keys : Iterable[K], optional
            Keys to retrieve the values for.
            If `None`, return all values in the container.
        """
        if keys is None:
            return list(self.data.values())
        return [self.data[k] for k in keys]

    def get_subset(self, keys: Iterable[K]) -> Self:
        """
        Get a subset of the data.

        Arguments
        ---------
        keys : Iterable[K]
            Keys included in the subset.

        Returns
        -------
        Container
            Container with the subset of data.
        """
        subset_data = {k: v for k, v in self.data.items() if k in keys}
        return self.__class__(subset_data, key_type=self.key_type, value_type=self.value_type)

    def filter_on_keys(self, predicate: Callable[[K], bool]) -> Self:
        """
        Filter the data for a subset of keys which satisfy a predicate.

        Arguments
        ---------
        predicate : Callable[[K], bool]
            Function to filter the keys.

        Returns
        -------
        Container
            Container with the items whose keys satisfy the predicate.

        Examples
        --------
        Filter the units in based on the brain area to which they belong:

        >>> def predicate(unit: Unit) -> bool:
        ...     return unit.area == "PFC"
        ...
        >>> data = {Unit("avo052a-d1"): 1, Unit("lemon052a-b2"): 2}
        >>> container = Container(data, key_type=Unit, value_type=int)
        >>> container.filter_on_keys(predicate)
        >>> container.data
        {Unit("avo052a-d1"): 1}
        """
        return self.get_subset([k for k in self.data.keys() if predicate(k)])

    def filter_on_values(self, predicate: Callable[[V], bool]) -> Self:
        """
        Filter the data for a subset of values which satisfy a predicate.

        Arguments
        ---------
        predicate : Callable[[V], bool]
            Function to filter the values.

        Returns
        -------
        Container
            Container with the items whose values satisfy the predicate.

        Examples
        --------
        Filter units in a container based on a threshold for the firing rate:

        >>> def predicate(rate: float) -> bool:
        ...     return rate > 1
        ...
        >>> data = {Unit("avo052a-d1"): 1.0, Unit("lemon052a-b2"): 2.0}
        >>> container = Container(data, key_type=Unit, value_type=float)
        >>> container.filter_on_values(predicate)
        Container({Unit("lemon052a-b2"): 2})
        """
        return self.get_subset([k for k, v in self.data.items() if predicate(v)])

    def apply(self, func: Callable[[V], R], **kwargs: Any) -> "Container[K, R]":
        """
        Apply a function to all values across keys, optionally with additional keyword arguments.

        Arguments
        ---------
        func : Callable[[V, Any], R]
            Function to apply to each value in the container. The function should take the value as
            a first argument and additional keyword arguments.
        **kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        Container[K, R]
            New container with transformed data.

        Examples
        --------
        Add a number to all values in a `Container` of string keys and integer values:

        >>> def add(data: int, increment: int) -> int:
        ...     return data + increment
        ...
        >>> container = Container({"a": 1, "b": 2}, key_type=str, value_type=int)
        >>> new_container = container.apply(add, increment=10)
        >>> print(new_container.data)
        {"a": 11, "b": 12}

        Implementation
        --------------
        To create a new instance, call `Container` instead of `self.__class__` to ensure that the
        new types can be set. Otherwise, the types would be inherited from the current instance,
        which are `Type[V]` for the values instead of `Type[R]`.

        Signature of the function (`Callable[[V], R]`):

        - It imposes a single positional argument: the value of type `V`.
        - It allows to pass any additional arguments as keyword arguments `**kwargs`.
        """
        # Apply the function to all values
        result_data = {k: func(v, **kwargs) for k, v in self.data.items()}
        # Determine the type of the result values
        _, result_type = self.find_types(result_data)
        # Create a new container with new data
        return Container(result_data, key_type=self.key_type, value_type=result_type)

    def __getattr__(self, method_name: str):
        """
        Proxy method calls to the values in the container.

        Arguments
        ---------
        method_name : str
            Name of the method to call on each value object.

        Returns
        -------
        method_proxy : Callable[..., Container]
            Function that applies the specified method to all values and returns a new container
            with the transformed results.

        Raises
        ------
        AttributeError
            If the method does not exist in the value type.

        Examples
        --------
        Define a class with a method to transform the value:

        >>> class ExampleValue:
        ...     def __init__(self, value: int):
        ...         self.value = value
        ...
        ...     def transform(self, factor: int) -> int:
        ...         return self.value * factor
        ...
        ...     def to_str(self) -> str:
        ...         return str(self.value)

        Call the `transform` method on a container via dot syntax, as if it were applied to each
        value individually:

        >>> data = {1: ExampleValue(1), 2: ExampleValue(2)}
        >>> container = Container(data, key_type=int, value_type=ExampleValue)
        >>> transformed_container = container.transform(factor=2)
        >>> print(transformed_container.data)
        {1: 2, 2: 4}
        >>> print(transformed_container.value_type)
        <class 'int'>

        Apply the `to_str` method to get another type of container:

        >>> string_container = container.to_str()
        >>> print(string_container.data)
        {1: '1', 2: '2'}
        >>> print(string_container.value_type)
        <class 'str'>
        """
        # Ensure the method exists on the value type
        if not hasattr(self.value_type, method_name):
            raise AttributeError(f"'{self.value_type.__name__}' has no attribute '{method_name}'")

        # Return a callable that applies the method to all values
        def method_proxy(*args, **kwargs):
            # Apply the method to all values
            result_data = {
                key: getattr(value, method_name)(*args, **kwargs)
                for key, value in self.data.items()
            }
            # Determine the type of the result values
            _, result_type = self.find_types(result_data)
            # Create a new container with new data
            return Container(result_data, key_type=self.key_type, value_type=result_type)

        return method_proxy

    @staticmethod
    def find_types(data: Dict[Q, R]) -> Tuple[Type[Q], Type[R]]:
        """
        Determine the type of the keys and values in a dictionary.

        Arguments
        ---------
        data : Dict[K, Any]
            Dictionary of data whose keys (resp. values) have the same type.

        Returns
        -------
        Tuple[Type[Q], Type[R]]
            Type of the keys and values in the dictionary.

        Raises
        ------
        TypeError
            If the data is empty, no types can be determined.

        Implementation
        --------------
        The types are based on the of the first value, assuming that all keys (resp. values) have
        the same type.
        """
        if data:
            first_key = next(iter(data.keys()))
            key_type = type(first_key)
            first_value = next(iter(data.values()))
            value_type = type(first_value)
        else:
            raise TypeError("Cannot determine types for an empty dictionary")
        return key_type, value_type


class UnitsContainer(Container[Unit, V]):
    """
    Container for population data.

    Class Attributes
    ----------------
    KEY_TYPE : Type[Unit]
        Type of the keys in the container.

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

    def __init__(self, *args, value_type: Type[V], **kwargs) -> None:
        """Fix `key_type` to `Units` and allow dynamic `value_type`."""
        super().__init__(*args, key_type=self.KEY_TYPE, value_type=value_type, **kwargs)

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
        Type of the keys in the container.

    Examples
    --------
    Initialize an `ExpCondContainer` with a dictionary of integer values:

    >>> exp_conds_to_values = {ExpCondition("a"): 1, ExpCondition("b"): 2}
    >>> container = ExpCondContainer(exp_conds_to_values, value_type=int)

    """

    KEY_TYPE = ExpCondition

    def __init__(self, *args, value_type: Type[V], **kwargs) -> None:
        """Fix `key_type` to `ExpCondition` and allow dynamic `value_type`."""
        super().__init__(*args, key_type=self.KEY_TYPE, value_type=value_type, **kwargs)

    @property
    def exp_conditions(self) -> List[ExpCondition]:
        """Get the experimental conditions in the container."""
        return list(self.data.keys())
