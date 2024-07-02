#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates.exp_cond` [module]

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

from mtcdb.coordinates.base_coord import Coordinate
from mtcdb.core_objects.exp_cond import Task, Context, Stimulus


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
    values: npt.NDArray[np.str_]
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
    In subclasses, the class attribute ``condition`` has to be overridden
    to match the specific condition type.

    See Also
    --------
    :class:`mtcdb.coordinates.base_coord.Coordinate`
    :mod:`mtcdb.core_objects.exp_cond`
    """
    condition: Type[C]

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)

    def __repr__(self):
        counts = self.count_by_lab()
        return f"<{self.__class__.__name__}> : {len(self)} samples, {counts}."

    @staticmethod
    def build_labels(n_smpl: int, cnd: C) -> npt.NDArray[np.str_]:
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
        values: npt.NDArray[np.str_]
            Condition coordinate filled with the label of the condition.
        """
        return np.full(n_smpl, cnd.value, dtype=np.str_)

    def replace_label(self, old: C, new: C) -> npt.NDArray[np.str_]:
        """
        Replace one label by another one in the condition coordinate.
        
        Parameters
        ----------
        old, new: C
            Conditions corresponding to the initial and new labels.

        Returns
        -------
        values: npt.NDArray[np.str_]
            Coordinate with updated condition labels.
        """
        values = self.values.copy()
        values[values == old.value] = new.value
        return values

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
            return {cnd: np.sum(self.values == cnd.value) for cnd in self.condition.get_options()}


class CoordTask(CoordExpCond[Task]):
    """
    Coordinate labels for tasks in which measurements were performed.

    See Also
    --------
    :class:`mtcdb.core_objects.exp_cond.Task`
    :class:`mtcdb.coordinates.CoordExpCond`
    """
    condition: Type[Task] = Task

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)


class CoordContext(CoordExpCond[Context]):
    """
    Coordinate labels for contexts in which measurements were performed.

    See Also
    --------
    :class:`mtcdb.core_objects.exp_cond.Context`
    :class:`mtcdb.coordinates.CoordExpCond`
    """
    condition: Type[Context] = Context

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)


class CoordStim(CoordExpCond[Stimulus]):
    """
    Coordinate labels for stimuli associated with the measurements.

    See Also
    --------
    :class:`mtcdb.core_objects.exp_cond.Stimulus`
    :class:`mtcdb.coordinates.CoordExpCond`
    """
    condition: Type[Stimulus] = Stimulus

    def __init__(self, values: npt.NDArray[np.str_]):
        super().__init__(values=values)
