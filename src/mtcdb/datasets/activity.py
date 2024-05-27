#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.dataset.activity` [module]

Class representing data structures for the neuronal activity.

Classes
-------
RawSession
    Raw data for one unit in one recording session.
"""

import numpy as np
from numpy.typing import NDArray

from mtcdb.constants import DATA_PATH, SMPL_RATE


class RawSession:
    """
    Raw data for one unit in one recording session.

    For each unit (neuron), raw data consists in its *spiking times*
    during all the trials of one recording session.
    This data is stored in a numpy array of floats with shape ``(2, nspikes)``
    under the format ``[[trial numbers], [spiking times]]``.
    
    **Spiking Times** (second row)
    Each element corresponds to the time (in seconds) of one spike emitted during the session. 
    Each time is *relative* to the beginning of the trial in which is occured 
    (i.e. time origin is resetted at 0 in each trial).
    
    **Trial Numbers** (first row)
    Each element corresponds to the index of one trial.
    Trial indices are stored in ascending order,
    from 1 to the number of trials in the session, 
    with contiguous duplicates (e.g. ``1111122223333...``).

    .. warning::
        Trial indices start at 1, not 0 (in contrast to Python indexing).
    
    .. note:: 
        Trial indices are stored under type ``float``, 
        because of the constrained homogeneous types in numpy arrays.
    
    Attributes
    ----------
    unit_id: str
        Unit's identifier.
    session_id: str
        Session's identifier.
    n_trials: int
        Number of trials in the session.
    sampling_rate: float
        Sampling time for the recording, in seconds.
    path_data: str 
        Base directory where the data files are stored.
    data: NDArray[np.float64]
        A numpy array of shape (2, nspikes) where the first row represents
        trial numbers and the second row represents spiking times.
    """
    data_dir: str = DATA_PATH
    """Base directory for data files (class attribute)."""

    def __init__(self, 
                 unit_id:str, session_id:str, 
                 sampling_rate:float = SMPL_RATE):
        """
        Constructor - Instantiate a RawSession object for one unit in one recording session.

        Load the raw data (if possible) and extract its basic properties.

        Parameters
        ----------
        unit_id: str
        session_id: str
        sampling_rate: float
            Default: :obj:`mtcdb.constants.SMPL_RATE`
        """
        self.unit_id = unit_id
        self.session_id = session_id
        self.path_data = self.get_path()
        self.data = self.load()
        self.spikes = self.data[1]
        self.trials = self.data[0]
        self.n_trials = max(self.trials)
        self.sampling_rate = sampling_rate
    
    def get_path(self) -> str:
        """Construct the path for the data file."""
        return f"{RawSession.data_dir}/{self.unit_id}/raw/{self.session_id}.npy"

    def load(self) -> NDArray[np.float64]:
        """
        Load the raw data from a ``.npy file``, if possible.

        Returns
        -------
        data: NDArray[np.float64]
            Shape: ``(2, nspikes)``
        
        Raises
        ------
        FileNotFoundError
            If the data file is not found in the data directory.
            An empty array is created with the right shape for consistency.
        """
        try:
            data = np.load(self.path_data)
        except FileNotFoundError:
            print(f"Data file not found in {self.path_data}")
            data = np.empty((2, 0))
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Unit {self.unit_id!r}, Session {self.session_id!r}"
