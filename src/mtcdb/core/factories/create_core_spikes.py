"""
`core.factories.create_core_spikes` [module]

Classes
-------
FactoryCoordSlot
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `create` method in `FactoryCoordSlot`
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------

from typing import List, Tuple, cast

import numpy as np


from core.factories.base_factory import Factory
from core.data_components.core_data import CoreData
from core.coordinates.exp_structure_coord import CoordSlot
from core.coordinates.time_coord import CoordTimeEvent
from core.data_structures.spike_times import SpikeTimesRaw
from core.data_structures.trials_properties import TrialsProperties
from core.attributes.exp_structure import Session, Slot
from core.composites.base_container import Container


class FactoryCoordSlot(Factory[Tuple[CoreData, CoordSlot]]):
    """
    Create the coordinate representing the slots in which spikes occur.

    Products: `CoreData`, `CoordSlot`

    - Inputs: Spike times and trials properties.
    - Output: Spiking times and coordinate marking the slots in which the spike times occurred.

    Methods
    -------
    create (required)
    """

    PRODUCT_CLASSES = Tuple[CoreData, CoordSlot]

    def create(
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
            spikes_aligned[idx_spikes] -= start  # type: ignore[misc]
            # Disabled: "Result type of - incompatible in assignment."
            # Reason: In-place operation with `DataComponent` (subclass of `np.ndarray`) causes type
            # mismatch between the result of the operation (numpy array) and the original object
            # (DataComponent).
            coord[idx_spikes] = slot
        return spikes_aligned, coord
