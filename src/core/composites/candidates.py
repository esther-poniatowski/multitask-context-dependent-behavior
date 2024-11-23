#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.exclude` [module]

Classes
-------
`Candidates`

"""
from collections import UserList
from typing import Iterable, Callable, TypeVar, Generic, List

T = TypeVar("T")
"""Type variable representing the type of elements in the candidate set."""


class Candidates(UserList, Generic[T]):
    """
    Utilities to exclude elements from a set from various criteria.

    Arguments
    ---------
    candidates : Iterable[T]
        Elements from which some members might be excluded. Input candidates can be provided under
        the form of any iterable, but are converted to a list for internal processing.

    Attributes
    ----------
    data : List[T]
        Internal list containing the candidate set.

    Methods
    -------
    `exclude`
    `filter`
    `filter_by_associated`

    See Also
    --------
    `collections.UserList`: Inherit from this class to provide list-like behavior.
    """

    def __init__(self, candidates: Iterable[T]) -> None:
        super().__init__(candidates)

    def to_list(self) -> List[T]:
        """
        Return the candidate set as a list.

        Returns
        -------
        List[T]
            Candidate set as a list.
        """
        return self.data

    def exclude(self, intruders: Iterable) -> None:
        """
        Exclude a set of intruders from the candidate set.

        Arguments
        ---------
        intruders : Iterable
            Elements to exclude from the candidate set.

        Examples
        --------

        >>> candidates = Candidates([1, 2, 3, 4, 5])
        >>> candidates.exclude(intruders=[2, 4])
        >>> candidates.get()
        [1, 3, 5]
        """
        self.data = [element for element in self.data if element not in intruders]

    def filter(self, predicate: Callable) -> None:
        """
        Exclude elements that do not satisfy a given predicate.

        Arguments
        ---------
        predicate : Callable
            Function that takes a single argument and returns a boolean, indicating whether an
            element satisfies the condition.

        Examples
        --------

        >>> candidates = Candidates([1, 2, 3, 4, 5])
        >>> candidates.filter(lambda x: x % 2 == 0)
        >>> candidates.get()
        [2, 4]
        """
        self.data = [element for element in self.data if predicate(element)]

    def filter_by_associated(self, values: Iterable, predicate: Callable) -> None:
        """
        Exclude elements based on a predicate applied to associated values.

        Arguments
        ---------
        values : Iterable
            Values associated with the candidates, in the same order.
        predicate : Callable
            Function that takes a single argument and returns a boolean, indicating whether an
            associated value satisfies the condition.

        Examples
        --------

        >>> candidates = Candidates(['a', 'b', 'c', 'd'])
        >>> candidates.filter_by_associated([1, 2, 3, 4], lambda x: x % 2 == 0)
        >>> candidates.get()
        ['b', 'd']
        """
        self.data = [element for element, value in zip(self.data, values) if predicate(value)]
