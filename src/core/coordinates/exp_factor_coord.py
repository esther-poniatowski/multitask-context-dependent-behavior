#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.exp_condition` [module]

Coordinates for labelling experimental conditions.

Classes
-------
`CoordExpFactor` (Generic)
`CoordTask`
`CoordAttention`
`CoordCategory`
`CoordStimulus`
`CoordBehavior`
`CoordEventDescription`
"""

from typing import TypeVar, Type, Optional, Union, Dict, Self, overload

import numpy as np

from core.coordinates.base_coord import Coordinate
from core.entities.exp_factors import (
    Task,
    Attention,
    Stimulus,
    Category,
    Behavior,
    EventDescription,
    ExpFactor,
)


ExpFactorType = TypeVar("ExpFactorType", bound=ExpFactor)
"""Generic type variable for experimental conditions. Used to keep a generic type for the
experimental factor while specifying the data type of the coordinate labels."""


class CoordExpFactor(Coordinate[np.str_, ExpFactorType]):
    """
    Coordinate labels representing one experimental condition among Task, Attention, Stimulus.

    Class Attributes
    ----------------
    ENTITY : Type[ExpFactor]
        Subclass of `Entity` corresponding to the type of experimental condition which is
        represented by the coordinate.
    DTYPE : Type[np.str_]
        Data type of the condition labels, always string. The `np.str_` dtype is equivalent to the
        `np.unicode_` dtype. It encompasses strings in fixed-width.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any], np.str_]
        Labels for the condition associated with each measurement.

    Methods
    -------
    `count_by_lab`
    `build_labels`
    `replace_label`

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    `core.entities.exp_factors`
    """

    ENTITY: Type[ExpFactorType]
    DTYPE = np.str_
    SENTINEL: str = ""

    def __repr__(self):
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{cnd!r}: {n}" for cnd, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    @classmethod
    def build_labels(cls, n_smpl: int, cnd: ExpFactor) -> Self:
        """
        Build basic labels filled with a *single* condition.

        Parameters
        ----------
        n_smpl : int
            Number of samples, i.e. of labels.
        cnd : ExpFactor
            Condition which corresponds to the single label.

        Returns
        -------
        values : Self
            Labels coordinate filled a single condition.
        """
        values = np.full(n_smpl, cnd)
        return cls(values)

    def replace_label(self, old: ExpFactor, new: ExpFactor) -> Self:
        """
        Replace one label by another one in the condition coordinate.

        Parameters
        ----------
        old, new : C
            Conditions corresponding to the initial and new labels.

        Returns
        -------
        value : Self
            Coordinate with updated condition labels.
        """
        new_coord = self.copy()
        new_coord[new_coord == old] = new
        return new_coord

    @overload
    def count_by_lab(self, cnd: ExpFactor) -> int: ...

    @overload
    def count_by_lab(self) -> Dict[ExpFactor, int]: ...

    def count_by_lab(self, cnd: Optional[ExpFactor] = None) -> Union[int, Dict[ExpFactor, int]]:
        """
        Count the number of samples in one condition or all conditions.

        Parameters
        ----------
        cnd : str, optional
            Condition to count in the coordinate.
            If None, give the number of samples in each condition.

        Returns
        -------
        n_smpl : Union[int, Dict[C, int]]
            If ``cnd`` is specified, number of samples matching this condition.
            Otherwise, number of samples in each condition.

        Implementation
        --------------
        The set of valid values for the condition is accessed by ``self.condition.get_options()``.
        """
        if cnd is not None:
            return np.sum(self == cnd)
        else:
            options = self.ENTITY.get_options()
            return {self.ENTITY(cnd): np.sum(self == cnd) for cnd in options}


class CoordTask(CoordExpFactor[Task]):
    """
    Coordinate labels for tasks.
    """

    ENTITY = Task


class CoordAttention(CoordExpFactor[Attention]):
    """
    Coordinate labels for attentional states.
    """

    ENTITY = Attention


class CoordCategory(CoordExpFactor[Category]):
    """
    Coordinate labels for categories.
    """

    ENTITY = Category


class CoordStimulus(CoordExpFactor[Stimulus]):
    """
    Coordinate labels for stimuli.
    """

    ENTITY = Stimulus


class CoordBehavior(CoordExpFactor[Behavior]):
    """
    Coordinate labels for behavioral choices of the animals.
    """

    ENTITY = Behavior


class CoordEventDescription(Coordinate[np.str_, EventDescription]):
    """
    Coordinate labels for event descriptions.
    """

    ENTITY = EventDescription
