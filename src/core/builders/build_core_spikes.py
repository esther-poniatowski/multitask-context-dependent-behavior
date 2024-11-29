#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_core_spikes` [module]

Classes
-------
SpikeTrainsBuilder
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `build` method in `NeuronalActivityBuilder`
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------

from typing import List, Tuple, cast

import numpy as np


from core.builders.base_builder import Builder
from core.data_structures.core_data import CoreData
from core.coordinates.exp_structure_coord import CoordSlot
from core.coordinates.time_coord import CoordTimeEvent
from core.data_structures.spike_times import SpikeTrains, SpikeTimesRaw
from core.data_structures.trials_properties import TrialsProperties
from core.attributes.exp_structure import Session, Recording, Block, Slot
from core.composites.base_container import Container
from core.attributes.brain_info import Unit
from core.constants import SMPL_RATE


class SpikeTrainsBuilder(Builder[SpikeTrains]):
    """
    Build the matrix gathering the spiking times of a single unit across all its recording sessions,
    from the raw spike times and trials properties in multiple individual sessions.

    Product: `SpikeTrains`

    - Inputs: List of spike times (flat structure) for a single unit in several recording sessions.
    - Output: Single array of spike times (flat structure) across all the sessions.

    Methods
    -------
    build (implementation of the base class method)
    """

    PRODUCT_CLASS = SpikeTrains

    def __init__(
        self,
    ) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters

    def build(
        self,
        spikes: Container[Session, SpikeTimesRaw],
        trials_properties: TrialsProperties,
        unit: Unit,
        smpl_rate: float = SMPL_RATE,
    ) -> SpikeTrains:
        """
        Implement the base class method.

        Arguments
        ---------
        spikes : List[SpikeTimesRaw]
            Spiking times of one unit in one or several session(s).
            Length: number of sessions.
            Shape of each element: ``(n_spikes,)``, with ``n_spikes`` the number of spikes in the
            considered session.
        trials_properties : TrialsProperties
            Metadata about the *trials* in *one or several* session(s).

        Returns
        -------
        spike_trains : SpikeTrains
            Spiking times of one unit across all the sessions.
            Shape: ``(n_sessions, n_spikes_tot)``, with ``n_sessions`` the number of sessions and
            ``n_spikes_tot`` the total number of spikes across all the sessions.
        """
        # Build the coordinate marking the slots in the spike times
        spikes_aligned, coord_slot = SlotCoordBuilder().build(spikes, trials_properties)
        return SpikeTrains(unit=unit, smpl_rate=smpl_rate, data=spikes_aligned, slot=coord_slot)


class SlotCoordBuilder(Builder[Tuple[CoreData, CoordSlot]]):
    """
    Build the coordinate representing the slots in which spikes occur.

    Product: `CoordSlot`

    - Inputs: Spike times and trials properties.
    - Output: Coordinate marking the slots in the spike times.

    Methods
    -------
    build (required)

    """

    PRODUCT = Tuple[CoreData, CoordSlot]

    def build(
        self,
        spikes: SpikeTimesRaw | Container[Session, SpikeTimesRaw],
        trials_properties: TrialsProperties,
    ) -> Tuple[CoreData, CoordSlot]:
        """
        Implement the base class method.

        Arguments
        ---------
        spikes : SpikeTimesRaw | Container[Session, SpikeTimesRaw]
            Spiking times of one unit in one or several session(s).
            Length: number of sessions.
            Shape of each element: ``(n_spikes,)``, with ``n_spikes`` the number of spikes in the
            considered session.
        trials_properties : TrialsProperties
            Metadata about the *trials* in *one or several* session(s).

        Returns
        -------
        spikes_aligned : CoreData
            Spiking times of one unit across all the sessions, relative to the beginning of their
            slot. Shape: ``(n_spikes_tot,)``, with ``n_spikes_tot`` the total number of spikes
            across all the sessions.
        coord_slot : CoordSlot
            Coordinate marking the slots in the spike times across all the sessions.
            Shape: ``(n_spikes_tot,)`` (idem).

        See Also
        --------
        `CoordSlot`
        """
        # Convert the input if single data structure for uniform processing
        if isinstance(spikes, SpikeTimesRaw):
            spikes = Container({Session(""): spikes}, key_type=Session, value_type=SpikeTimesRaw)
        # Determine the sessions order to align with the concatenated spikes
        sessions = spikes.keys()
        ordered_sessions = Session.order(*sessions)
        # Compute coordinates for every session
        spikes_in_sessions: List[CoreData] = []
        coords_in_sessions: List[CoordSlot] = []
        for session in ordered_sessions:
            # Extract spiking times
            raw_data = spikes[session].get_data()
            # Extract trials properties in the session
            trials_in_session = trials_properties.get_session(session)
            trials_slot = cast(CoordSlot, trials_in_session.get_coord("slot"))
            trials_start = cast(CoordTimeEvent, trials_in_session.get_coord("t_start"))
            trials_end = cast(CoordTimeEvent, trials_in_session.get_coord("t_end"))
            # Compute the coordinate for the session
            spk, coord = self.mark_slots(raw_data, trials_slot, trials_start, trials_end)
            spikes_in_sessions.append(spk)  # shape: (n_spikes_session,)
            coords_in_sessions.append(coord)  # shape: (n_spikes_session,)
        # Concatenate the coordinates across the sessions
        spikes_aligned = CoreData(np.concatenate(spikes_in_sessions))  # shape: (n_spikes_tot,)
        coord_slot = CoordSlot(np.concatenate(coords_in_sessions))
        return spikes_aligned, coord_slot

    @staticmethod
    def mark_slots(
        spikes: CoreData,
        trials_slot: CoordSlot,
        trials_start: CoordTimeEvent,
        trials_end: CoordTimeEvent,
    ) -> Tuple[CoreData, CoordSlot]:
        """
        Mark the slots in the spike times with the corresponding trial index in a single session.

        Arguments
        ---------
        spikes : CoreData
            Spiking times of one unit in a single session. Shape: ``(n_spikes,)``.
        trials_slot, trials_start, trials_end : CoordSlot, CoordTimeEvent, CoordTimeEvent
            Metadata about the *trials* in a *single* session.

        Notes
        -----
        Coordinates of interest among the trials properties:

        - "slot": Index of the slot of the trial relative to the block to which it belongs.
        - "t_start": Start time of each trial relative to its block.
        - "t_end": End time of each trial relative to its block.

        Returns
        -------
        spikes_aligned : CoreData
            Spiking times of one unit in the session, relative to the beginning of their slot.
            Shape: ``(n_spikes,)``, with ``n_spikes`` the number of spikes in the session.
        coord_slot : CoordSlot
            Coordinate indicating the slots in which each spike occurs.
            Shape: ``(n_spikes,)`` (idem).

        Implementation
        --------------
        - Iterate over the slots (trials) referenced in the trials properties.
        - For each slot, retrieve its time boundaries.
        - Identify the spikes occurring in this time interval.
        - Shift the spiking times to be relative to the beginning of the slot.
        - Assign the slot index to those spikes in the coordinate.

        Reference for time intervals: In the `spikes` array, the spiking times are relative to the
        beginning of the *block*, which indeed matches the time boundaries of the slots in the
        coordinates `t_start` and `t_end`.

        See Also
        --------
        `CoordSlot`
        """
        n_spikes = len(spikes)
        spikes_aligned = spikes.copy()
        coord = CoordSlot.from_shape(n_spikes)
        for slot, start, end in zip(trials_slot, trials_start, trials_end):
            start = cast(float, start)
            idx_spikes = np.where((spikes >= start) & (spikes < end))[0]
            spikes_aligned[idx_spikes] -= start
            coord[idx_spikes] = slot
        return spikes_aligned, coord
