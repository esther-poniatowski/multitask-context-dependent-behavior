#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.data_structures.datasets` [module]
"""
from abc import ABC, abstractmethod
import copy
import numpy as np
import numpy.typing as npt
from types import MappingProxyType
from typing import Tuple, Dict
from typing import Type, TypeVar, Generic

from mtcdb.constants import SMPL_RATE, T_BIN, D_PRE, D_POST, D_STIM, D_PRESHOCK, D_SHOCK, SMOOTH_WINDOW
from mtcdb.io_handlers.path_managers.impl import PathManager, RawSpkTimesPath, SpikesTrainsPath
from mtcdb.io_handlers.loaders.impl import LoaderPKL, LoaderNPY
from mtcdb.io_handlers.savers.impl import SaverPKL


T = TypeVar('T')
"""
Type variable representing the type of raw data set in the generic Data class.
"""

class Data(ABC, Generic[T]):
    """
    Base class for data structures.
    
    Class Attributes
    ----------------
    dims: Tuple[str]
        Names of the dimensions. 
    coords: Dict[str, str]
        Repertoire of coordinates associated to each dimension.
        Keys: Dimension name.
        Values: Coordinate name, i.e. attribute under which 
                it is stored in the data structure.
    path_manager: Type[PathManager]
        Path manager class to build paths for data files.
        Pick the right path manager subclass in each data subclass.
    saver: Type[Saver], default=SaverPKL
        Saver class to save the data to files in a specific format.
    loader: Type[Loader], default=LoaderPKL
        Loader class to load the data from files.
    tpe : Type, default='Data'
        Type of the loaded data (parameter of the loader).
    
    Attributes
    ----------
    data: npt.NDArray
        Actual data values to analyze.
    dims: List[str]
        Override the class attribute to associate dimensions to axes in the data array.
        This object is mutable to accommodate transformations of the data which affect the dimensions,
        such as transpositions (dimension permutation) or reshaping (dimension fusion).
        The name of an axis is determined by the corresponding dimension name :
        ``dims[axis]``
        The axis of each dimension is determined by the order in the list : 
        ``dims.index(name)``
    coord2dim: MappingProxyType[str, str]
        Mapping of the coordinates to the dimensions of the data.
        Keys: Coordinate names.
        Values: Dimension names.
        This attribute is immutable to ensure that the structure remains consistent.
    shape: Tuple[int]
        Shape of the data array (delegated to the numpy array).
    n: MappingProxyType[str, int]
        Number of elements along each dimension.
        Keys: Dimension names.
        Values: Number of elements.
        
    Methods
    -------
    __init__
    __repr__
    copy
    path
    load
    save
    
    Notes
    -----
    Several attributes are read-only to ensure the integrity of the data structure.
    Those attributes are : 
    - data
    - dims
    - coord2dim

    Implementation
    --------------
    Information associated to each dimension is specified in dictionaries
    whose keys are dimensions names (instead of tuples in which the order matters).
    This is relevant to to abstract away from the order of the dimensions
    which might change in data transformations.

    Examples
    --------
    Access the number of time points in data with a time dimension:
    >>> data.n['time']
    
    See Also
    --------
    :meth:`npt.NDArray.setflags`
        Set the flags of the numpy array to make it read-only.
    """
    dims: Tuple[str]
    coords: Dict[str, str]
    path_manager: Type[PathManager]
    saver: Type[Saver] = SaverPKL
    loader: Type[Loader] = LoaderPKL
    tpe : Type[T] = 'Data'
    
    def __init__(self, data: npt.NDArray) -> None:
        self._data = data
        self._data.setflags(write=False)
        self.coord2dim = MappingProxyType({coord: dim for dim, coord in self.coords.items()})
        self.shape = self.data.shape
        self.n = MappingProxyType({dim: shape for dim, shape in zip(self.dims, self.shape)})

    @property
    def data(self) -> npt.NDArray:
        """
        Read-only access to the actual data values to analyze.
        
        Returns
        -------
        npt.NDArray
        """
        return self._data
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> Dims: {self.dims}\n Coords: {list(self.coords.values())}"

    def copy(self) -> 'Data':
        """
        Copy the data structure.

        Returns
        -------
        Data

        See Also
        --------
        :meth:`copy.deepcopy`
        """
        return copy.deepcopy(self)
    
    @abstractmethod
    @property
    def path(self) -> Path:
        """
        Build the path to the file containing the data.

        Overridden in data subclasses to provide the required arguments 
        to the path manager from the attributes of the data structure.

        Returns
        -------
        Path
        """
        return self.path_manager.get_path()
    
    def load(self) -> 'Data':
        """
        Retrieve data from a file.

        Notes
        -----
        The raw data is recovered in the type specified by :obj:`tpe`.
        If needed, transform it in an instance of the data structure.
        By default, with pickle, the data is directly recovered 
        as an object corresponding to the data structure.

        Returns
        -------
        Data
        """
        data = self.loader(self.path, self.tpe).load() # chain loader methods
        data = self.__class__(data) # call constructor
        return data
    
    def save(self) -> None:
        """
        Save the data instance.

        Notes
        -----
        If needed, the data structure should be transformed
        in the format expected by the saver.
        By default, with pickle, the object can be saved directly
        without any transformation.
        """
        self.saver(self.path, self.data).save()

    def sel(self, **kwargs) -> 'Data':
        """
        Select data along specific coordinates.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            Keys : Coordinate names.
            Values : Selection criteria (single value, list or slice).

        Returns
        -------
        Data
            New data structure containing the selected data.

        Example
        -------
        Select time points between 0 and 1 second:
        >>> data.sel(time=slice(0, 1))
        Select trials in task 'PTD':
        >>> data.sel(task='PTD')
        Select trials for stimuli 'R' and 'T':
        >>> data.sel(stim=['R', 'T'])
        Select error trials only:
        >>> data.sel(error=True)
        Select along multiple coordinates:
        >>> data.sel(time=slice(0, 1), task='PTD', stim=['R', 'T'])
        """
        # Unpack the names of the coordinates present in the data structure
        coord_attrs = list(self.coord2dim.keys())
        # Initialize True masks for each dimensions to select all elements by default
        masks = {dim: np.ones(shape, dtype=bool) for dim, shape in zip(self.dims, self.shape)}
        # Update masks with the selection criteria on each coordinate
        for name, label in kwargs.items():
            if name in self.coord_attrs:
                dim = self.coord2dim[name] # dimension to which the coordinate applies
                coord = getattr(self, name) # coordinate object
                # Create a boolean mask depending on the target label type
                if isinstance(label, list):
                    mask = np.isin(coord.values, label)
                elif isinstance(label, slice):
                    mask = np.zeros(coord.values.size, dtype=bool)
                    mask[label] = True
                elif isinstance(label, (int, str, bool)):
                    mask = (coord.values == label)
                else:
                    raise TypeError(f"Invalid type for label '{label}'.")
                # Apply this mask to the corresponding axis
                masks[dim] = masks[dim] & mask
        # Convert the boolean masks to integer indices
        indices = {dim: np.where(mask)[0] for dim, mask in masks.items()}
        # Combine masks along all dimensions (meshgrid)
        mesh = np.ix_(*[indices[dim] for dim in self.dims]) # order masks by their corresponding dimensions
        # Select the data
        new_data = self.data[mesh]
        # Select the coordinates
        new_coords = {name: getattr(self, name) for name in coord_attrs}
        for name, _ in kwargs:
            if name in self.coord_attrs:
                new_coords[name] = coord[indices[self.coord2dim[name]]]
        # Instantiate a new data structure with the selected data,
        # by unpacking the new coordinates dictionary
        raise NotImplementedError("Method not implemented yet.")
        return self.__class__(new_data, **new_coords)


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
