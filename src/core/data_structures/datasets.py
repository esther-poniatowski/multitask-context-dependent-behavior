#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.data_structures.datasets` [module]
"""
import numpy as np
import numpy.typing as npt
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, Dict
from typing import Type, TypeVar, Generic

from core.constants import SMPL_RATE, T_BIN, D_PRE, D_POST, D_STIM, D_PRESHOCK, D_SHOCK, SMOOTH_WINDOW
from utils.io.path_managers.impl import PathManager, RawSpkTimesPath, SpikesTrainsPath
from utils.io.loaders.base import Loader
from utils.io.savers.base import Saver
from utils.io.loaders.impl import LoaderPKL, LoaderNPY
from utils.io.savers.impl import SaverPKL
from core.data_structures.base import Data
from core.coordinates import CoordBlock


class RawSpkTimes(Data):
    """
    Raw spiking times for one unit (neuron) in one session of the experiment.

    Attributes
    ----------
    data: npt.NDArray[np.float64]
        Spiking times for the unit in the session (in seconds).
        Times are reset at 0 in each block.
    block: CoordBlock
        Coordinate for the sessions' block in which each spike occurred.
        Ascending order, from 1 to the number of blocks in the session,
        with contiguous duplicates (e.g. ``1111122223333...``).
    unit_id: str
        Unit's identifier.
    session_id: str
        Session's identifier.
    smpl_rate: float
        Sampling time for the recording (in seconds).
        Default: :obj:`mtcdb.constants.SMPL_RATE`

    Notes
    -----
    This data structure represents the entry point of the data analysis.
    Therefore, it tightly reflects to the raw data, without additional pre-processing.
    It is not meant to be saved, therefore no saver is defined.

    Warning
    -------
    Each spiking time (in seconds) is *relative* to the beginning of the block
    of trials in which is occurred (i.e. time origin is reset at 0 in each block).
    """
    dims: Tuple[str] = ("time",)
    coords: Dict[str, str] = {"time": "block"}
    path_manager: Type[PathManager] = RawSpkTimesPath
    loader: Type[Loader] = LoaderNPY
    tpe : Type[T] = npt.NDArray[np.float64]

    def __init__(self,
                 data: npt.NDArray[np.float64],
                 block: 'CoordBlock',
                 unit_id: str,
                 session_id: str,
                 smpl_rate: float = SMPL_RATE
                 ) -> None:
        super().__init__(data=data)
        self.block = block
        self.unit_id = unit_id
        self.session_id = session_id
        self.smpl_rate = smpl_rate

    def __repr__(self) -> str:
        return f"Unit {self.unit_id}, Session {self.session_id}\n" + super().__repr__()

    @property
    def path(self) -> Path:
        return self.path_manager.get_path(self.unit_id, self.session_id)

    def load(self) -> 'RawSpkTimes':
        """
        Retrieve data from a file.

        Notes
        -----
        The raw data of one neuron in one session is stored in a
        ``npt.NDArray`` of floats with shape ``(2, nspikes)``.
        ``raw[0]`` : Block of trials in which each spike occurred.
        ``raw[1]`` : Spiking times.
        In this initial array, trial indices are stored under type ``float``,
        because of the constraint homogeneous types in numpy arrays.
        Here they are converted to integers.

        Returns
        -------
        Data
        """
        raw = super().load()
        block = raw[0].astype(np.int64)
        t_spk = raw[1]
        return self.__class__(t_spk, block, self.unit_id, self.session_id, self.smpl_rate)

    @property
    def n_blocks(self) -> int:
        """
        Number of blocks in the session.

        Returns
        -------
        int
        """
        if self.trials.size != 0: # avoid ValueError in max()
            return self.block.max()
        else: # empty array
            return 0

    def get_block(self, blk: int) -> 'RawSpkTimes':
        """
        Extract the spiking times which occurred in one block of trials.

        Parameters
        ----------
        blk: int
            Block number, comprised between 1 and the total number of blocks.

        Returns
        -------
        RawSpkTimes
            Spiking times for the unit in the specified block.
        """
        mask = self.block == blk
        data = self.data[mask]
        block = self.block[mask]
        return self.__class__(data, block, self.unit_id, self.session_id, self.smpl_rate)


class SpikesTrains(Data):
    """
    Spikes trains for one unit in a set of trials of the experiment.

    Attributes
    ----------
    data: npt.NDArray[np.float64]
        Spiking times of the unit in different trials (in seconds).
        Shape: ``(n_trials, n_spk_max)``
        with ``n_spk_max`` the maximal number of spikes across trials.
    rec: npt.NDArray[str]
        Recording number of the session to which each trial belongs.
    block: npt.NDArray[int]
        Block to which each trial belongs.
    slot: npt.NDArray[int]
        Slot of the block forming each trial.
    task: npt.NDArray[str]
        Task condition in each trial.
    ctx: npt.NDArray[str]
        Context condition in each trial.
    stim: npt.NDArray[str]
        Stimulus condition in each trial.
    error: npt.NDArray[bool]
        Behavioral outcome in each trial (possibly True in NoGo trials only)
    t_on: npt.NDArray[float]
        Time of the stimulus onset in each trial (in seconds).
    t_off: npt.NDArray[float]
        Time of the stimulus offset in each trial (in seconds).
    t_end: npt.NDArray[float]
        Time of the end of the trial, i.e. duration (in seconds).
        It might be longer than the time of the last spike.

    Warning
    -------
    Each "trial" corresponds to one slot in one block in one recording session.
    In each trial, the spiking times are reset at 0 relative to the beginning of the *slot*,
    instead of the beginning of the block.

    Notes
    -----
    The shape ``n_spk_max`` of the data array is determined by the maximal number of spikes across trials.
    If the ith trial contains ``n_spk`` spikes, then the row ``data[i]`` associated to this trial has :
    - ``n_spk`` cells storing the spiking times (from index 0 to ``n_spk-1``)
    - ``n_spk_max - n_spk`` remaining cells filled with ``NaN``.

    This data structure represents an intermediary step between :class:`RawSpkTimes` and :class:`FiringRates`.
    It centralizes information about spikes and trials to avoid repeating expensive computations.
    Indeed, most coordinates have to be built by parsing the raw ``expt_events`` files of the sessions.

    This data structure can be used for the following purposes :
    - Visualizing raster plots before firing rate are computed.
    - Selecting trials and units for the final analyses.
      To assess several criteria, additional processing is required on each separate trial.

    New coordinates will be added to store the filtering metrics.
    """
    dims: Tuple[str] = ("time", "trial")
    coords: Dict[str, str] = {"trial": ["rec", "block", "slot", "task", "ctx", "stim", "error"],}
    path_manager: Type[PathManager] = SpikesTrainsPath
    loader: Type[Loader] = LoaderPKL
    tpe : Type[T] = 'Data'
    saver: Type[Saver] = SaverPKL

    def __init__(self,
                 data: npt.NDArray[np.float64],) -> None:
        super().__init__(data=data)


class FiringRatesUnit(Data):
    """
    Firing rates for one unit in a set of trials.

    Attributes
    ----------
    unit_id: str
        Unit's identifier.
    sessions: List[str]
        All the session identifiers.
    data: npt.NDArray[np.float64]
        Firing rates for the unit in all the trials of all its recording sessions.
        Shape: ``(n_tpts, n_trials)``.
    n_trials: int
        Total number of trials across all the sessions.
    n_tpts: int
        Number of time points in a trial's time course.
    t_max: float
        Total duration the firing rate time course.
    t_on, t_off: float
        Times of the stimulus onset and offset.
    t_bin: float
        Time bin for the firing rate time course.
    smooth_window: float
        Smoothing window size (in seconds).
    """
    def __init__(self,
                 unit_id:str,
                 t_bin:float = T_BIN,
                 d_pre:float = D_PRE,
                 d_post:float = D_POST,
                 d_stim:float = D_STIM,
                 d_pre_shock:float = D_PRESHOCK,
                 d_shock:float = D_SHOCK,
                 smooth_window:float = SMOOTH_WINDOW):
        super().__init__(self.loader, self.saver, self.empty_shape)
        self.unit_id = unit_id
        self.path_data = self.get_path()
        self.data = self.load()
        self.n_tpts, self.n_trials = self.data.shape
        self.d_pre = d_pre
        self.d_post = d_post
        self.d_stim = d_stim
        self.d_pre_shock = d_pre_shock
        self.d_shock = d_shock
        self.t_on = 0
        self.t_off = self.t_on + d_stim
        self.t_bin = t_bin
        self.t_max = self.n_tpts*t_bin
        self.smooth_window = smooth_window


class FiringRatesPop(Data):
    """
    Firing rates for a pseudo-population in a set of pseudo-trials.

    Attributes
    ----------
    data: npt.NDArray[np.float64]
        Firing rates for all the units in the considered trials.
        Shape: ``(n_units, n_tpts, n_trials)``.
    n_units: int
        Total number of units in the population.
    n_trials: int
        Total number of trials under consideration.
    n_tpts: int
        Number of time points in a trial's time course.
    t_max: float
        Total duration the firing rate time course.
    t_on, t_off: float
        Times of the stimulus onset and offset.
    t_bin: float
        Time bin for the firing rate time course.
    smooth_window: float
        Smoothing window size (in seconds).
    """

    def __init__(self):
        pass


'''


The second decision is about the access to the coordinate objects from the data structure.
Indeed, the user may want to access some attributes of the coordinate object directly via the data structure,
without having to call the coordinate object itself.

The third decision is about the indexing of the data structure with the coordinate objects.
I might want to implement a kind of "sel" method as in xarray,
to select data along a specific coordinate.
This method would return a new data structure with the selected data,
and all its other attributes should be updated accordingly.

'''
