#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.exp_condition` [module]

Coordinates for labelling experimental conditions.

Classes
-------
`CoordExpCond` (Generic)
`CoordTask`
`CoordCtx`
`CoordStim`
"""

from typing import TypeVar, Type, Optional, Union, Dict, Self, overload

import numpy as np

from coordinates.base_coord import Coordinate
from core.entities.exp_condition import Task, Context, Stimulus


ExpCondition = TypeVar("ExpCondition", Task, Context, Stimulus)
"""Generic type variable for experimental conditions entities."""


class CoordExpCond(Coordinate[np.str_, ExpCondition]):
    """
    Coordinate labels representing one experimental condition among Task, Context, Stimulus.

    Class Attributes
    ----------------
    ENTITY : Type[ExpCondition]
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
    :class:`core.coordinates.base_coord.Coordinate`
    :mod:`core.entities.exp_condition`
    """

    ENTITY: Type[ExpCondition]
    DTYPE = np.str_
    SENTINEL: str = ""

    def __repr__(self):
        counts = self.count_by_lab()
        format_counts = ", ".join([f"{cnd!r}: {n}" for cnd, n in counts.items()])
        return f"<{self.__class__.__name__}>: {len(self)} samples, {format_counts}."

    @classmethod
    def build_labels(cls, n_smpl: int, cnd: ExpCondition) -> Self:
        """
        Build basic labels filled with a *single* condition.

        Parameters
        ----------
        n_smpl : int
            Number of samples, i.e. of labels.
        cnd : ExpCondition
            Condition which corresponds to the single label.

        Returns
        -------
        values : Self
            Labels coordinate filled a single condition.
        """
        values = np.full(n_smpl, cnd.value)
        return cls(values)

    def replace_label(self, old: ExpCondition, new: ExpCondition) -> Self:
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
        new_coord[new_coord == old.value] = new.value
        return new_coord

    @overload
    def count_by_lab(self, cnd: ExpCondition) -> int: ...

    @overload
    def count_by_lab(self) -> Dict[ExpCondition, int]: ...

    def count_by_lab(
        self, cnd: Optional[ExpCondition] = None
    ) -> Union[int, Dict[ExpCondition, int]]:
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
            return np.sum(self == cnd.value)
        else:
            options = self.ENTITY.get_options()
            return {self.ENTITY(cnd): np.sum(self == cnd) for cnd in options}


class CoordTask(CoordExpCond[Task]):
    """
    Coordinate labels for tasks.
    """

    ENTITY = Task


class CoordCtx(CoordExpCond[Context]):
    """
    Coordinate labels for contexts.
    """

    ENTITY = Context


class CoordStim(CoordExpCond[Stimulus]):
    """
    Coordinate labels for stimuli.
    """

    ENTITY = Stimulus


class CoordEventDescription(Coordinate[np.str_]):
    """
    Coordinate labels for event descriptions.

    Each element is a string which can comprise several event descriptions separated by commas.

    Examples:

    - ``'PreStimSilence , TORC_448_06_v501 , Reference'``
    - ``'TRIALSTART'``

    Attributes
    ----------
    values: npt.NDArray[np.str_]
        Labels for the event descriptions associated with each measurement.

    Notes
    -----
    No specific entity is associated with event descriptions.

    See Also
    --------
    :class:`core.coordinates.base_coord.Coordinate`
    """

    DTYPE = np.str_
    SENTINEL: str = ""
