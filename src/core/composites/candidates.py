#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.exclude` [module]

Classes
-------
`Excluder`

"""

from typing import List, Iterable, Callable, TypeVar, Generic

T = TypeVar("T")
"""Type variable representing the type of elements in the candidate set."""


class Candidates(Generic[T]):
    """
    Utilities to exclude elements from a set from various criteria.

    Attributes
    ----------
    candidates : List[T]
        Elements from which some members might be excluded. Input candidates can be provided under
        the form of any iterable, but are converted to a list for internal processing.

    Methods
    -------
    `exclude`
    `filter`
    """

    def __init__(self, candidates: Iterable[T]) -> None:
        self.candidates = list(candidates)

    def get(self) -> List[T]:
        """Get the candidate set, possibly after exclusions."""
        return self.candidates

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
        self.candidates = [element for element in self.candidates if element not in intruders]

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
        self.candidates = [element for element in self.candidates if predicate(element)]
