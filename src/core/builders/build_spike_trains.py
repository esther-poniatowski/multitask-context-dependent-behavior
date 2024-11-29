#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.builders.build_spike_trains` [module]

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

from typing import List

import numpy as np

from core.builders.base_builder import Builder
from core.data_structures.core_data import CoreData
from core.data_structures.spike_times import SpikeTrains, SpikeTimesRaw
from core.data_structures.trials_properties import TrialsProperties
from core.attributes.exp_structure import Session, Recording, Block, Slot
from core.composites.base_container import Container


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
        spikes_in_sessions: Container[Session, SpikeTimesRaw],
        trials_properties: TrialsProperties,
    ) -> SpikeTrains:
        """
        Implement the base class method.

        Parameters
        ----------
        spikes_in_sessions : Container[Session, SpikeTrains]
            Spiking times of one unit in each recording session (dictionary-like container).
        trials_properties : TrialsProperties
            Metadata about the trials across the experiment. It is used to retrieve the time
            boundaries of each slot in each block.

        Returns
        -------
        product : SpikeTrains
            Shape: ``(n_spk_tot,)``, with ``n_spk_tot`` the total number of spikes across all the
            sessions.

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
        """
        # Initialize the core data array with empty values
        n_spk_tot = self.eval_spikes_number(spikes_in_sessions.list_values())
        shape = (n_spk_tot,)
        data = CoreData(shape)
        # Determine the order of the sessions
        order_sessions = Session.order(*spikes_in_sessions.keys())
        # Format sessions one by one
        for session in order_sessions:
            raw_data = spikes_in_sessions[session]
            spikes_in_blocks = self.segment_in_blocks(raw_data)
            for block, spikes in spikes_in_blocks.items():
                spikes_in_slots = self.segment_in_slots(spikes)
                for slot, spikes_slot in enumerate(spikes_in_slots):
                    idx_trial = trials_properties.get_index(session, block, slot)
                    self.fill_data(data, spikes_slot, idx_trial)
            self.fill_data(data, spikes)
        # Fill trial by trial
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

    @staticmethod
    def eval_spikes_number(spikes_in_sessions: List[SpikeTimesRaw]) -> int:
        """
        Evaluate the total number of spikes across all the sessions.

        Parameters
        ----------
        spikes_in_sessions : List[SpikeTimesRaw]
            See the argument :ref:`spikes_in_sessions`.

        Returns
        -------
        n_spk_tot : int
            Total number of spikes across all the sessions.
        """
        return sum(spikes.n_spikes for spikes in spikes_in_sessions)

    @staticmethod
    def mark_slots(spikes, trials_properties):
        """
        Mark the slots in the spike times with the corresponding trial index.

        Returns
        -------
        CoordSlot
            Coordinate marking the slots in the spike times.

        See Also
        --------
        `CoordSlot`
        """
        pass

    @staticmethod
    def

    @staticmethod
    def segment_in_blocks(raw_data: SpikeTimesRaw) -> Container[Block, CoreData]:
        """
        Segment the spiking times in one session into blocks of trials.
        """
        blocks = sorted(np.unique(raw_data.get_coord("block")))
        spikes_in_blocks = Container(
            {block: raw_data.get_block(block) for block in blocks},
            key_type=Block,
            value_type=CoreData,
        )
        return spikes_in_blocks

    @staticmethod
    def segment_in_slots(raw_data: SpikeTimesRaw) -> List[CoreData]:
        """
        Segment the spiking times in one block into slots of trials.
        """
        slots = sorted(np.unique(raw_data.get_coord("slot")))
        spikes_in_slots = [raw_data.get_slot(slot) for slot in slots]
        return spikes_in_slots
