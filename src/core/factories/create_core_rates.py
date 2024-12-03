#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.factories.create_core_rates` [module]

Classes
-------
FactoryFiringRates

Notes
-----
This class operates on both trials properties and spike trains, which are aligned by their
positional information allowing to reconstruct each trial.

The main operations performed by the current class are:

1. Extracting the relevant epochs of the spikes trains for each trial.
2. Processing those epochs to compute the firing rates.
3. Filling the core data with the result.
"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `create` method in `FactoryFiringRates`
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------

from typing import List

import numpy as np

from core.factories.base_factory import Factory
from core.data_components.core_data import CoreData
from core.data_components.core_dimensions import Dimensions
from core.data_structures.spike_times import SpikeTrains
from core.data_structures.trials_properties import TrialsProperties
from core.coordinates.trial_analysis_label_coord import CoordPseudoTrialsIdx


class FactoryFiringRates(Factory[CoreData]):
    """
    Create the matrix gathering the activity of a single unit in selected pseudo-trials, from the raw
    spike times and trials properties in multiple sessions.

    Product: `CoreData`

    - Inputs: List of spike times (flat structure) for a single unit in (several) recording
      sessions.
    - Output: Trial-based representation of its activity.

    Attributes
    ----------

    Methods
    -------

    Methods
    -------
    create (implementation of the base class method)
    """

    PRODUCT_CLASSES = CoreData

    def __init__(self) -> None:
        # Call the base class constructor: declare empty product and internal data
        super().__init__()
        # Store configuration parameters

    def create(
        self,
        spikes: SpikeTrains,
        trials_properties: TrialsProperties,
        pseudo_trials_idx: CoordPseudoTrialsIdx,
    ) -> CoreData:
        """
        Implement the base class method.

        Parameters
        ----------
        spikes : SpikeTrains
            Spiking times of one unit across the experiment.
        trials_properties : TrialsProperties
            Properties of the trials in the experiment.
        pseudo_trials_idx : CoordPseudoTrialsIdx
            Coordinate of the trials indices to select in the global data set to reconstruct
            pseudo-trials for the unit.
            Shape: ``(n_pseudo,)``.

        Returns
        -------
        product : CoreData
            Firing rates of the unit in pseudo-trials, i.e. actual values to analyze.
        """
        extractor = SpikesExtractor()
        converter = FiringRatesConverter()
        # Initialize core data
        n_folds, n_pseudo = pseudo_trials_idx.shape
        shape = (n_folds, n_pseudo)
        data = CoreData.from_shape(
            shape,
            dims=Dimensions("trials"),
        )
        # Fill the data structure, trial by trial
        for i_final, i_init in enumerate(pseudo_trials_idx):
            # Extract all the spikes occurring in the trial
            recording, block, slot = trials_properties.get_info(
                i_init, "recording", "block", "slot"
            )
            spikes_in_trial = spikes.get_trial(recording, block, slot)
            # Extract the relevant epochs from the trial's spikes train
            spikes_in_epochs = extractor.process(spikes_in_trial)
            # Convert those epochs to firing rates
            firing_rates = converter.process(spikes_in_epochs)
            # Fill the core data with the result
            data[i_final] = firing_rates
        return data

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
        `Factory.initialize_core_data`
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
