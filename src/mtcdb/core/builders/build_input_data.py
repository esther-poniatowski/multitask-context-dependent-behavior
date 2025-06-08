"""
`core.pipelines.format_input_data` [module]

Classes
-------
BuilderSpikeTrains
"""
from typing import Type, Dict, List, Any
from dataclasses import dataclass, field

import numpy as np

from core.constants import SMPL_RATE
from core.builders.base_builder import Builder
from core.data_structures.spike_times import SpikeTrains, SpikeTimesRaw
from core.attributes.brain_info import Area, Training, Unit
from core.attributes.exp_structure import Session
from core.composites.exp_conditions import ExpCondition
from core.composites.coordinate_set import CoordinateSet
from core.composites.base_container import Container
from core.coordinates.exp_factor_coord import CoordExpFactor
from core.data_structures.trials_properties import TrialsProperties


class BuilderSpikeTrains(Builder[SpikeTrains]):
    """
    Gather and format the spiking times of a single unit across all its recording sessions, from the
    raw spike times and trials properties in multiple individual sessions.

    Product: `SpikeTrains`

    - Inputs: List of spike times (flat structure) for a single unit in several recording sessions.
    - Output: Single array of spike times (flat structure) across all the sessions.

    Methods
    -------
    build (implementation of the base class method)
    """

    PRODUCT_CLASSES = SpikeTrains

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
