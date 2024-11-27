#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_raster` [module]

Classes
-------
`RasterBuilder`

Notes
-----

"""
# pylint: disable=missing-function-docstring

from typing import List, Tuple, Optional, Any, TypeAlias

import numpy as np

from core.builders.base_builder import Builder[DataStructure]
from core.coordinates.base_coord import Coordinate
from core.coordinates.exp_factor_coord import CoordTask, CoordAttention
from core.coordinates.exp_structure_coord import CoordRecNum
from core.data_structures.core_data import CoreData
from core.data_structures.spike_times_raw import SpikeTimesRaw
from core.data_structures.spike_trains import SpikeTrains
from core.processors.preprocess.align_spikes import SpikesAligner

MetaDataSession: TypeAlias = Any


class SpikeTrainsBuilder(Builder[DataStructure][List[SpikeTimesRaw], SpikeTrains]):
    """
    Build a `SpikeTrains` data structure from the raw spike times for a single unit in multiple
    sessions.

    - Inputs: List of raw spike times (flat structure) for a single unit in (several) recording
      sessions.
    - Output: Trial-based representation of the spike times into spike trains.

    Detailed inputs and outputs are documented in the `build` method.

    Class Attributes
    ----------------
    product_class : type
        See the base class attribute.
    TMP_DATA : Tuple[str]
        See the base class attribute.

    Processing Attributes
    ---------------------
    spikes_per_session : List[SpikeTimesRaw]
        Raw spike times for the unit in each session.
    metadata_per_session : List[MetaDataSession]
        Metadata for each session, used to build the coordinates.
    sessions_boundaries : List[Tuple[int, int]]
        Start and end indices of the trials for each session in the final data structure.

    Methods
    -------
    `build` (implementation of the base class method)
    """

    product_class = SpikeTrains
    TMP_DATA = ("spikes_per_session", "metadata_per_session")

    def __init__(self) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters

        # Declare attributes to store inputs and intermediate results
        self.spikes_per_session: List[SpikeTimesRaw]
        self.metadata_per_session: List[MetaDataSession]
        self.sessions_boundaries: List[Tuple[int, int]]

    def build(
        self,
        unit_id: Optional[str] = None,
        smpl_rate: Optional[float] = None,
        spikes_per_session: Optional[List[SpikeTimesRaw]] = None,
        metadata_per_session: Optional[List[MetaDataSession]] = None,
        **kwargs,
    ) -> SpikeTrains:
        """
        Implement the base class method.

        Parameters
        ----------
        unit_id : str
            Identifier of the unit for which the spike trains are built.
        smpl_rate : float
            Sampling rate of the spike times (in Hz).
        spikes_per_session : List[SpikeTimesRaw]
            See the attribute `spikes_per_session`.
        metadata_per_session : List[MetaDataSession]
            See the attribute `metadata_per_session`.

        Returns
        -------
        product : SpikeTrains
            Data structure product instance.
        """
        assert (
            unit_id is not None
            and smpl_rate is not None
            and spikes_per_session is not None
            and metadata_per_session is not None
        )
        # Store inputs
        assert len(spikes_per_session) == len(metadata_per_session)
        self.spikes_per_session = spikes_per_session
        self.metadata_per_session = metadata_per_session
        # Preliminary set up
        self.sessions_boundaries = self.allocate_sessions_indices()
        # Initialize the data structure with its metadata (base method)
        self.initialize_data_structure(unit_id=unit_id, smpl_rate=smpl_rate)
        # Add core data values to the data structure
        data = self.construct_core_data()
        self.add_data(data)
        # Add coordinates in the data structure
        recnum = self.construct_new_coord("recnum", CoordRecNum)
        block = self.construct_preexisting_coord("block")
        slot = self.construct_preexisting_coord("slot")
        task = self.construct_new_coord("task", CoordTask)
        attn = self.construct_new_coord("attn", CoordAttention)
        categ = self.construct_preexisting_coord("categ")
        error = self.construct_preexisting_coord("error")
        t_on = self.construct_preexisting_coord("t_on")
        t_off = self.construct_preexisting_coord("t_off")
        t_warn = self.construct_preexisting_coord("t_warn")
        t_end = self.construct_preexisting_coord("t_end")
        self.add_coords(recnum=recnum, block=block, slot=slot)
        self.add_coords(task=task, attn=attn, categ=categ, error=error)
        self.add_coords(t_on=t_on, t_off=t_off, t_warn=t_warn, t_end=t_end)
        return self.get_product()

    # --- Shape and Dimensions ---------------------------------------------------------------------

    @property
    def n_sessions(self) -> int:
        """Number of sessions in which the unit was recorded."""
        return len(self.spikes_per_session)

    @property
    def n_trials(self) -> int:
        # TODO: Implement
        """Total number of trials across all the sessions. 1st dimension of the data product."""
        return sum(
            session_info.count_trials(session_info) for session_info in self.metadata_per_session
        )

    @property
    def n_spk_max(self) -> int:
        """Maximum number of spikes across trials and sessions. 2nd dimension of the data product."""
        # TODO: Implement
        return max(
            self.max_spikes_in_session(session_info, spike_times)
            for session_info, spike_times in zip(self.metadata_per_session, self.spikes_per_session)
        )

    # --- Preliminary Operations -------------------------------------------------------------------

    def allocate_sessions_indices(self) -> List[Tuple[int, int]]:
        """
        Set the start and end indices of the trials of each session in the final data structure.

        Those indices are used to ensure the consistency of the data between the core data and the
        coordinates for the trials dimension.

        Returns
        -------
        sessions_boundaries : List[Tuple[int, int]]
            Start and end indices of the trials for each session along the trials dimension in the
            final data structure.
            Length: ``n_sessions``. Each element is a tuple of two integers (start, end).
        """
        sessions_boundaries: List[Tuple[int, int]] = []
        start = 0
        for session_info in self.metadata_per_session:
            end = start + session_info.count_trials()
            self.sessions_boundaries.append((start, end))
            start = end
        return sessions_boundaries

    def max_spikes_in_session(
        self, session_info: MetaDataSession, spike_times: SpikeTimesRaw
    ) -> int:
        """
        Determine the maximal number of spikes across trials in a single session.

        Arguments
        ---------
        session_info : MetaDataSession
            Metadata of the session.
        spike_times : SpikeTimesRaw
            Raw spike times for the unit in the session.

        Returns
        -------
        n_spk_max : int
            Maximal number of spikes across trials in the session.
        """
        n_spk_max = 0
        for b in range(spike_times.n_blocks):
            for s in range(session_info.count_slots_in_block(b)):
                n_spk = len(spike_times.get_spikes_in_slot(b, s))
                n_spk_max = max(n_spk_max, n_spk)
        return n_spk_max

    # --- Construct Core Data ----------------------------------------------------------------------

    def construct_core_data(self) -> CoreData:
        """
        Construct the core data array containing the spike trains of the unit in the trials.

        Returns
        -------
        data : CoreData
            Data array containing the spike times.
            Shape: ``(n_trials, n_spk_max)``.
            .. _data:

        Raises
        ------
        ValueError
            If the number of blocks in the spike times does not match the number of blocks in the
            session.
            If the number of trials in the spike times does not match the number of trials in the
            session.

        Implementation
        --------------
        1. Initialize the core data array with empty values.
        2. Process each session independently. The spiking data of the session and the session's
           metadata are considered in parallel to ensure their consistency.
           1. Segment the spike times in the session into blocks of trials only once, to avoid
              redundant operations.
           2. Iterate over the trials mentioned in the session's metadata and retrieve the
              information required to extract the spikes in this trial (slot): block, start and end
              time of the trial. This approach ensures to consider trials in the appropriate order
              to match the coordinates constructed from the sessions' metadata.
           3. Fill the core data with the spiking times of the considered trial. To ensure the
              correspondence of the trials' indices between the coordinates and the core data,
              The index where data is added is the index of the slot being processed in addition to
              an offset corresponding to the start index of the session in the final data structure.

        See Also
        --------
        `Builder[DataStructure].initialize_core_data`
            Initialize the core data array with empty values (parent method).
        `SpikesAligner.slice_epoch`
            Extract spiking times within one epoch and reset them relative to the epoch start.
        `TrialsProperties.iter_trials`
            Provide the relevant metadata to describe the trials in the session.
        """
        aligner = SpikesAligner()
        shape = (self.n_trials, self.n_spk_max)
        data = self.initialize_core_data(shape)  # parent method
        for (start, end), spike_times, session_info in zip(
            self.sessions_boundaries, self.spikes_per_session, self.metadata_per_session
        ):
            spikes_in_blocks = self.segment_in_blocks(spike_times)
            n_b_spk, n_b_sess = len(spikes_in_blocks), session_info.n_blocks
            if n_b_spk != n_b_sess:
                raise ValueError(f"Blocks mismatch: {n_b_spk} (spikes) != {n_b_sess} (session).")
            i = None  # declare before entering the loop for subsequent check
            for i, (block, slot, t_start, t_end) in enumerate(session_info.iter_trials()):
                spikes_slot = aligner.slice_epoch(spikes_in_blocks[block], t_start, t_end)
                idx_trial = start + i  # offset by the start index of the session
                self.fill_data(data, spike_times=spikes_slot, idx_trial=idx_trial)
            if i != end - 1:  # check that all trials have been filled at the end of the loop
                raise ValueError(f"Trials mismatch: {i} (spikes) != {end} (session).")
        return data

    def segment_in_blocks(self, spike_times: SpikeTimesRaw) -> List[np.ndarray]:
        """
        Segment the spike times in the session into blocks of trials.

        Arguments
        ---------
        spike_times : SpikeTimesRaw
            Raw spike times for the unit in the session.

        Returns
        -------
        spikes_in_blocks : List[np.ndarray]
            Spike times in each block of trials. Length: number of blocks in the session.

        See Also
        --------
        `SpikeTimesRaw.get_block`
            Extract the spiking times which occurred in one block of trials.
        """
        return [spike_times.get_block(b) for b in range(spike_times.n_blocks)]

    def fill_data(self, data: CoreData, spike_times, idx_trial: int) -> None:
        """
        Fill a single trial of the data array with the spiking times in one slot.

        Arguments
        ---------
        data : CoreData
            See the return :ref:`data`.
        spike_times : np.ndarray
            Spiking times in the slot. Shape: ``(n_spk,)``, where ``n_spk`` is the number of spikes,
            which is less or equal to the maximal number of spikes across all the trials.
        idx_trial : np.ndarray
            Index of the trial in the data array along the trials dimension (first dimension).

        Notes
        -----
        Along the second dimension, the data array remains padded with NaN values for trials with
        less than the maximal number of spikes.
        """
        data[idx_trial, : len(spike_times)] = spike_times

    # --- Construct Coordinates --------------------------------------------------------------------

    def construct_new_coord(self, name, coord_class: type[Coordinate]) -> Coordinate:
        """
        Construct a new coordinate for the trials dimension containing session's information.

        Each session is associated with one recording number, one task and one attentional state, and yields
        as many number of elements as the total number of trials it contains.

        Parameters
        ----------
        name : str
            Name of the coordinate to build.
        coord_class : type[Coordinate]
            Class of the coordinate to build. Used to convert the array of values into a coordinate
            object of the appropriate type.

        Returns
        -------
        coord : Coordinate
            Coordinate for the trial dimension in the data structure product.
        """
        values = [getattr(session_info, name) for session_info in self.metadata_per_session]
        counts = [session_info.count_trials() for session_info in self.metadata_per_session]
        arrays = [(np.repeat(v, c)) for v, c in zip(values, counts)]
        coord = coord_class(np.concatenate(arrays))
        self.check_coord_size(coord, name)
        return coord

    def construct_preexisting_coord(self, coord_name: str) -> Coordinate:
        """
        Construct a coordinate for the trials dimension form the corresponding pre-existing
        coordinates in the sessions's metadata.

        Parameters
        ----------
        coord_name : str
            Name of the coordinate to build, matching the attribute name in the metadata.

        Returns
        -------
        coord : Coordinate
            Coordinate for the trial dimension in the data structure product.

        Notes
        -----
        - The order of the sessions is consistent with the core data by concatenating the
          coordinates in the order of the sessions.
        - Concatenation is possible since coordinate objects are subclass of numpy arrays.
        - Concatenation is performed along the first axis (axis=0 by default) which results in a 1D
          array of shape ``(n_trials,)`` (sine all the coordinates are 1D arrays).
        - There is no need to specify the data type of the coordinate since it is inferred from the
          pre-existing coordinates.
        """
        coord = np.concatenate(
            [getattr(session_info, coord_name) for session_info in self.metadata_per_session]
        )
        self.check_coord_size(coord, coord_name)
        return coord

    def check_coord_size(self, coord: Coordinate, name: str) -> None:
        """
        Check that the size of a coordinate matches the total number of trials.

        Parameters
        ----------
        coord : Coordinate
            Coordinate to check.
        name : str
            Name of the coordinate.

        Raises
        ------
        ValueError
            If the dimension of the coordinate does not match the total number of trials expected
            when all the sessions are gathered.
        """
        if coord.size != self.n_trials:
            raise ValueError(
                f"Dimension mismatch ({name}): {coord.size} (coord) != {self.n_trials} (expected)."
            )
