#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.coordinates.time` [module]

Coordinate for labelling time stamps at which measurements were performed.

Classes
-------
`CoordTime`
`CoordTimeEvent`
"""

from typing import Optional, Self
import warnings

import numpy as np
from numpy.typing import ArrayLike

from coordinates.base_coord import Coordinate


class CoordTime(Coordinate[np.float64]):
    """
    Coordinate labels for time stamps at which measurements were performed.

    Class Attributes
    ----------------
    DTYPE : np.float64
        Data type of the time stamps, always float.
    METADATA : FrozenSet[str]
        Additional time markers.
    SENTINEL : float
        Sentinel value marking missing or unset time stamps, here `np.nan`.

    Attributes
    ----------
    values : npt.ndarray[Tuple[Any], np.float64]
        Time labels (in seconds), one dimensional.
        Homogeneous sequence starting from 0 and incremented by the time bin.
        Shape : ``(n_smpl,)`` with ``n_smpl`` the number of time stamps.
    t_bin : Optional[float]
        Time bin of the sampling (in seconds).
        None if the time stamps are not uniformly spaced or if the coordinate is empty.
    t_on : Optional[float]
        Time of stimulus onset (in seconds).
    t_off : Optional[float]
        Time of stimulus offset (in seconds).
    t_shock : Optional[float]
        Time of shock delivery (in seconds).

    Methods
    -------
    `validate`
    `eval_t_bin`
    `get_index`
    `build_labels`

    Notes
    -----
    No specific entity is associated with time.

    See Also
    --------
    `core.coordinates.base_coord.Coordinate`
    """

    # No ENTITY
    DTYPE = np.float64
    METADATA = frozenset(["t_on", "t_off", "t_shock", "t_bin"])
    SENTINEL: float = np.nan

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {len(self)} time points, bin = {self.t_bin} sec"

    def __new__(
        cls,
        values: ArrayLike,
        t_on: Optional[float] = None,
        t_off: Optional[float] = None,
        t_shock: Optional[float] = None,
        t_bin: Optional[float] = None,
    ):
        """Override the base method to pass metadata."""
        if t_bin is None:
            t_bin = cls.eval_t_bin(values)
        obj = Coordinate.__new__(cls, values, t_on=t_on, t_off=t_off, t_shock=t_shock, t_bin=t_bin)
        return obj

    @classmethod
    def validate(cls, values: ArrayLike, **kwargs) -> None:
        """
        Validate the time labels to ensure the existence of a time bin.

        Override the base method since no entity is associated with time.

        Raises
        ------
        UserWarning
            If a single time stamp is provided.
            If the time points are not uniformly spaced.
        """
        values = np.asarray(values)
        if len(values) < 2:  # at least two elements required for `diff`
            warnings.warn("Single time stamp, time bin not defined.")
        else:
            diffs = np.diff(values)  # shape : (n_smpl - 1,)
            if not np.allclose(diffs, diffs[0]):  # inhomogeneous sequence
                warnings.warn("Time points not uniformly spaced.")

    @classmethod
    def eval_t_bin(cls, values: ArrayLike) -> float:
        """
        Evaluate the time bin for the coordinate.

        Returns
        -------
        t_bin : float
            See the attribute `t_bin`.
            If only one time point is provided, the time bin cannot be defined and is evaluated to
            `np.nan`.
            Otherwise, it is evaluated from the distance between the two first time points.
        """
        values = np.asarray(values)
        if len(values) < 2:
            return np.nan
        else:
            return values[1] - values[0]

    def get_index(self, t: float) -> int:
        """
        Get the index of the closest value to a specific time in the sequence.

        Parameters
        ----------
        t : float
            Time point to index.

        Returns
        -------
        index : int
            Index of the closest value in the time sequence.
        """
        return int(np.argmin(np.abs(self.values - t)))

    @classmethod
    def build_labels(
        cls,
        n_smpl: Optional[int] = None,
        t_bin: Optional[float] = None,
        t_min: float = 0,
        t_max: Optional[float] = None,
    ) -> Self:
        """
        Build basic time labels from minimal parameters.

        Parameters
        ----------
        n_smpl : int, optional
            Number of time points to generate.
        t_bin : float, optional
            Time bin of the coordinate (in seconds).
        t_min, t_max : float, optional
            Time boundaries of the coordinate (in seconds).

        Returns
        -------
        coord : CoordTime
            Time coordinate instance.

        Notes
        -----
        Regardless of the provided parameters, the generated sequence of time points
        is always uniformly spaced and starts at ``t_min``.
        Different combinations of parameters allow to specify various creation constraints :

        - ``n_smpl``, ``t_bin``, (``t_min``) :
           -> ``n_smpl`` time points incremented by ``t_bin``.
        - ``n_smpl``, ``t_max``, (``t_min``) :
           -> ``n_smpl`` time points between ``t_min`` and ``t_max``.
        - ``t_bin``, ``t_max``, (``t_min``) :
           -> ``(t_max - t_min)/t_bin`` time points between ``t_min`` and ``t_max``.

        Implementation
        --------------
        To ensure the consistency of the time coordinate across methods, it is always build from the
        number of time points and the time bin. If ``n_smpl`` is not provided, it is computed as :
        ``n_smpl = int((t_max - t_min) / t_bin)`` If ``t_bin`` does not divide the interval,
        ``int()`` truncates the division, thus the last time point is not exactly ``t_max``, but the
        closest multiple of ``t_bin`` below ``t_max``.
        """
        if n_smpl is not None and t_bin is not None and t_max is None:
            pass  # no need to adjust the parameters
        elif n_smpl is not None and t_max is not None and t_bin is None:
            t_bin = (t_max - t_min) / n_smpl
        elif t_bin is not None and t_max is not None and n_smpl is None:
            n_smpl = int((t_max - t_min) / t_bin)
        else:
            raise ValueError("Invalid parameter combination.")
        values = np.arange(n_smpl) * t_bin + t_min
        return cls(values=values, t_bin=t_bin)


class CoordTimeEvent(Coordinate[np.float64]):
    """
    Coordinate labels for time stamps at which an experimental or behavioral event occurred.

    Examples: stimulus onset, stimulus offset, shock delivery...

    Attributes
    ----------
    values : np.ndarray[Tuple[Any], np.float64]
        Time labels (in seconds), one dimensional.
        Shape: ``(n_smpl,)``, number of samples in which events occurred.

    Notes
    -----
    No specific entity is associated with time events.
    """

    # No ENTITY
    DTYPE = np.float64
    SENTINEL: float = np.nan
