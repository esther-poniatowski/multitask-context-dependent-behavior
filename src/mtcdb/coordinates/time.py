#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.coordinates.time` [module]

Coordinate for labelling time points.

Classes
-------
:class:`CoordTime`
"""

from typing import Optional
import warnings

import numpy as np
import numpy.typing as npt

from mtcdb.coordinates.base_coord import Coordinate
from mtcdb.constants import TON, TOFF, TSHOCK


class CoordTime(Coordinate):
    """
    Coordinate labels for times at which measurements were performed.

    Attributes
    ----------
    values: npt.NDArray[np.float64]
        Time coordinate.
        Homogeneous sequence of time points (in seconds),
        starting from 0 and incremented by the time bin.
        Shape : ``(n_smpl,)`` with ``n_smpl`` the number of time points.
    tbin: Optional[float]
        Time bin of the coordinate (in seconds).
        None if the time points are not uniformly spaced.
    t_on: float
        Time of stimulus onset (in seconds).
    t_off: float
        Time of stimulus offset (in seconds).
    t_shock: float
        Time of shock delivery (in seconds).

    Methods
    -------
    :meth:`set_tbin`

    See Also
    --------
    :class:`mtcdb.coordinates.base_coord.Coordinate`
    """
    def __init__(self,
                 values: npt.NDArray[np.float64],
                 t_on: float = TON,
                 t_off: float = TOFF,
                 t_shock: float = TSHOCK):
        super().__init__(values=values)
        self.set_tbin()
        self.t_on = t_on
        self.t_off = t_off
        self.t_shock = t_shock

    def __repr__(self):
        return f"<{self.__class__.__name__}> : {len(self)} time points at {self.tbin} s bin."

    def set_tbin(self):
        """
        Recover the time bin from the coordinate values.
        
        Raises
        ------
        UserWarning
            If the time points are not uniformly spaced.
            No time bin can be computed.
        """
        diffs = np.diff(self.values) # shape : (n_smpl - 1,)
        if np.allclose(diffs, diffs[0]):
            self.tbin = diffs[0]
        else:
            warnings.warn("Time points not uniformly spaced.")
            self.tbin = None

    @staticmethod
    def build_labels(n_smpl: Optional[int] = None,
                     tbin: Optional[float] = None,
                     t_min: float = 0,
                     t_max: Optional[float] = None
                     ) -> npt.NDArray[np.float64]:
        """
        Build a time coordinate from scratch.

        Parameters
        ----------
        n_smpl: int, optional
            Number of time points to generate.
        tbin: float, optional
            Time bin of the coordinate (in seconds).
        t_min, t_max: float, optional
            Time boundaries of the coordinate (in seconds).

        Returns
        -------
        values: npt.NDArray[np.float64]
            Time points coordinate.

        Notes
        -----
        Regardless of the provided parameters, the generated sequence of time points 
        is always uniformly spaced and starts at ``t_min``. 
        Different combinations of parameters allow to specify various creation constraints :

        - ``n_smpl``, ``tbin``, (``t_min``) : 
           -> ``n_smpl`` time points incremented by ``tbin``.
        - ``n_smpl``, ``t_max``, (``t_min``) :
           -> ``n_smpl`` time points between ``t_min`` and ``t_max``.
        - ``tbin``, ``t_max``, (``t_min``) :
           -> ``(t_max - t_min)/tbin`` time points between ``t_min`` and ``t_max``.
        
        Implementation
        --------------
        To ensure the consistency of the time coordinate across methods,
        it is always build from the number of time points and the time bin.
        If ``n_smpl`` is not provided, it is computed as :
        ``n_smpl = int((t_max - t_min) / tbin)``
        If ``tbin`` does not divide the interval, ``int()`` truncates the division,
        thus the last time point is not exactly ``t_max``,
        but the closest multiple of ``tbin`` below ``t_max``.
        """
        if n_smpl is not None and tbin is not None and t_max is None:
            pass
        elif n_smpl is not None and t_max is not None and tbin is None:
            tbin = (t_max - t_min) / n_smpl
        elif tbin is not None and t_max is not None and n_smpl is None:
            n_smpl = int((t_max - t_min) / tbin)
        else:
            raise ValueError("Unexpected parameter combination.")
        return np.arange(n_smpl) * tbin + t_min
