#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates.exp_condition` [module]

Coordinates for labelling experimental conditions.

Classes
-------
:class:`CoordExpCond` (Generic)
:class:`CoordTask`
:class:`CoordContext`
:class:`CoordStim`
"""

from typing import TypeVar, Type, Generic, Optional, Union, Dict

import numpy as np
import numpy.typing as npt

from mtcdb.coordinates.base import Coordinate
from mtcdb.core_objects.exp_condition import Task, Context, Stimulus


C = TypeVar('C', Task, Context, Stimulus)
"""Generic type variable for experimental conditions."""


class CoordExpCond(Coordinate, Generic[C]):
    """
    Sub-class representing one experimental condition among :
    Task, Context, Stimulus.

    Class Attributes
    ----------------
    condition: Type[C]
        Reference to the class of the experimental condition object.
    
    Attributes
    ----------
    values: npt.NDArray[np.unicode_]
        Labels for the condition associated with each measurement.
        It contains the string values of the conditions 
        stored in the attribute ``value`` of the condition objects.

    Methods
    -------
    :meth:`count_by_lab`
    :meth:`replace_label`

    Implementation
    --------------
    All the conditions correspond to object types implemented in Enum classes.
    Here, the coordinates retain only the *string values* of the conditions.
    However, any interaction with the coordinate values should be done
    through condition objects (instead of strings themselves),
    to ensure the consistency of the data and types.
    In subclasses, the class attribute :attr:`condition` has to be overridden
    to match the specific condition type.

    See Also
    --------
    :class:`mtcdb.coordinates.base.Coordinate`
    :mod:`mtcdb.core_objects.exp_condition`
    """
    condition: Type[C]

    def __init__(self, values: npt.NDArray[np.unicode_]):
        super().__init__(values=values)

    def __repr__(self):
        counts = self.count_by_lab()
        return f"<{self.__class__.__name__}> : {len(self)} samples, {counts}."

    @staticmethod
    def build_labels(n_smpl: int, cnd: C) -> npt.NDArray[np.unicode_]:
        """
        Build a condition coordinate filled with a *single* label.
        
        Parameters
        ----------
        n_smpl: int
            Number of samples, i.e. of labels.
        cnd: C
            Condition which corresponds to the single label.
        
        Returns
        -------
        values: npt.NDArray[np.unicode_]
            Condition coordinate filled with the label of the condition.
        """
        return np.full(n_smpl, cnd.value, dtype=np.unicode_)

    def replace_label(self, old: C, new: C) -> 'CoordExpCond':
        """
        Replace one label by another one in the condition coordinate.
        
        Parameters
        ----------
        old, new: C
            Conditions corresponding to the initial and new labels.

        Returns
        -------
        values: npt.NDArray[np.unicode_]
            Coordinate with updated condition labels.
        """
        new_coord = self.copy()
        values = new_coord.values
        values[values == old.value] = new.value
        new_coord.values = values
        return new_coord

    def count_by_lab(self, cnd: Optional[C] = None
                     ) -> Union[int, Dict[C, int]]:
        """
        Count the number of samples in one condition or all conditions.
        
        Parameters
        ----------
        cnd: str, optional
            Condition to count in the coordinate.
            If None, give the number of samples in each condition.

        Returns
        -------
        n_smpl: Union[int, Dict[C, int]]
            If ``cnd`` is specified: Number of samples matching this condition.
            Otherwise: Number of samples in each condition.

        Implementation
        --------------
        The set of valid values for the condition is accessed by
        ``self.condition.get_options()`` (list of possible instances).
        """
        if cnd is not None:
            return np.sum(self.values == cnd.value)
        else:
            options = self.condition.get_options()
            return {self.condition(cnd): np.sum(self.values == cnd) for cnd in options}


class CoordTask(CoordExpCond[Task]):
    """
    Coordinate labels for tasks in which measurements were performed.

    See Also
    --------
    :class:`mtcdb.core_objects.exp_condition.Task`
    :class:`mtcdb.coordinates.CoordExpCond`
    """
    condition: Type[Task] = Task

    def __init__(self, values: npt.NDArray[np.unicode_]):
        super().__init__(values=values)


class CoordContext(CoordExpCond[Context]):
    """
    Coordinate labels for contexts in which measurements were performed.

    See Also
    --------
    :class:`mtcdb.core_objects.exp_condition.Context`
    :class:`mtcdb.coordinates.CoordExpCond`
    """
    condition: Type[Context] = Context

    def __init__(self, values: npt.NDArray[np.unicode_]):
        super().__init__(values=values)


class CoordStim(CoordExpCond[Stimulus]):
    """
    Coordinate labels for stimuli associated with the measurements.

    See Also
    --------
    :class:`mtcdb.core_objects.exp_condition.Stimulus`
    :class:`mtcdb.coordinates.CoordExpCond`
    """
    condition: Type[Stimulus] = Stimulus

    def __init__(self, values: npt.NDArray[np.unicode_]):
        super().__init__(values=values)
