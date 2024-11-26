#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.exp_factor_coord` [module]

Coordinates for labelling experimental factors.

Classes
-------
`CoordExpFactor` (Generic)
`CoordTask`
`CoordAttention`
`CoordCategory`
`CoordStimulus`
`CoordBehavior`
`CoordOutcome`
`CoordEventDescription`
"""

from typing import TypeVar, Type, Optional, Union, Dict, Self, overload

import numpy as np

from core.coordinates.base_coord import Coordinate
from core.attributes.exp_factors import (
    Task,
    Attention,
    Stimulus,
    Category,
    Behavior,
    ResponseOutcome,
    EventDescription,
    ExpFactor,
)


ExpFactorType = TypeVar("ExpFactorType", bound=ExpFactor)
"""Generic type variable for experimental factors. Used to keep a generic type for the
experimental factor while specifying the data type of the coordinate labels."""


class CoordExpFactor(Coordinate[np.str_, ExpFactorType]):
    """
    Coordinate labels representing one experimental factor.

    Class Attributes
    ----------------
    ENTITY : Type[ExpFactor]
        Subclass of `Attribute` corresponding to the type of experimental factor which is
        represented by the coordinate.
    DTYPE : Type[np.str_]
        Data type of the labels, always string. The `np.str_` dtype is equivalent to the
        `np.unicode_` dtype. It encompasses strings in fixed-width.

    Attributes
    ----------
    values : np.ndarray[Tuple[Any], np.str_]
        Labels for the factor associated with each measurement.

    Methods
    -------
    `count_by_lab`
    `build_labels`
    `replace_label`

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    `core.attributes.exp_factors`
    """

    ENTITY: Type[ExpFactorType]
    DTYPE = np.str_
    SENTINEL: str = ""

    def __repr__(self):
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{lab!r}: {n}" for lab, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    @classmethod
    def build_labels(cls, n_smpl: int, lab: ExpFactor) -> Self:
        """
        Build basic labels filled with a *single* factor.

        Parameters
        ----------
        n_smpl : int
            Number of samples, i.e. of labels.
        lab : ExpFactor
            Condition which corresponds to the single label.

        Returns
        -------
        values : Self
            Labels coordinate filled a single factor.
        """
        values = np.full(n_smpl, lab)
        return cls(values)

    def replace_label(self, old: ExpFactor, new: ExpFactor) -> Self:
        """
        Replace one label by another one in the factor coordinate.

        Parameters
        ----------
        old, new : C
            Conditions corresponding to the initial and new labels.

        Returns
        -------
        value : Self
            Coordinate with updated factor labels.
        """
        new_coord = self.copy()
        new_coord[new_coord == old] = new
        return new_coord

    @overload
    def count_by_lab(self, lab: ExpFactor) -> int: ...

    @overload
    def count_by_lab(self) -> Dict[ExpFactorType, int]: ...

    def count_by_lab(self, lab: Optional[ExpFactor] = None) -> Union[int, Dict[ExpFactorType, int]]:
        """
        Count the number of samples in one label or all labels.

        Parameters
        ----------
        lab : str, optional
            Condition to count in the coordinate.
            If None, give the number of samples in each label.

        Returns
        -------
        n_smpl : Union[int, Dict[C, int]]
            If ``lab`` is specified, number of samples matching this label.
            Otherwise, number of samples in each label.

        Implementation
        --------------
        The set of valid values for the label is accessed by ``self.ENTITY.get_options()``.
        """
        if lab is not None:
            return np.sum(self == lab)
        else:
            options = self.ENTITY.get_options()
            return {self.ENTITY(lab): np.sum(self == lab) for lab in options}


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


class CoordOutcome(CoordExpFactor[ResponseOutcome]):
    """
    Coordinate labels for the outcomes of the behavioral responses.
    """

    ENTITY = ResponseOutcome


class CoordEventDescription(Coordinate[np.str_, EventDescription]):
    """
    Coordinate labels for event descriptions.
    """

    ENTITY = EventDescription
