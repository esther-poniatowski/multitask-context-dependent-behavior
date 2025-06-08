"""
`core.attributes.analysis_attributes` [module]

Classes representing labels associated with trials for use in analytical workflows: fold
assignments, trial indices, or other analytical categorizations of experimental data.

Classes
-------
TrialAnalysisLabel
"""
from typing import Self

from core.attributes.base_attribute import Attribute


class TrialAnalysisLabel(int, Attribute[int]):
    """
    Any pos

    Define the `__new__` method to inherit from `int`.

    Methods
    -------
    is_valid (override the method from the base class `Attribute`)
    """

    MIN = 0

    def __new__(cls, value: int) -> Self:
        if cls.is_valid(value):  # method from the current subclass
            raise ValueError(f"Invalid value for {cls.__name__}: {value}")
        return super().__new__(cls, value)

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """
        Check if the value is a valid index or label.

        Override the method from the base class `Attribute`.
        """
        if value < cls.MIN or not isinstance(value, int):
            return False
        return True


class Fold(TrialAnalysisLabel):
    """
    Fold label for cross-validation.
    """


class TrialIndex(TrialAnalysisLabel):
    """
    Index of a trial in a data set.
    """
