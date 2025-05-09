#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.data_structures.firing_rates_pop` [module]

Classes
-------
FiringRatesPop
"""
from types import MappingProxyType

from core.data_components.core_dimensions import DimensionsSpec
from core.data_components.base_data_component import ComponentSpec
from core.data_components.core_metadata import MetaDataField
from core.data_components.core_data import CoreRates
from core.coordinates.base_coordinate import Coordinate
from core.coordinates.brain_info_coord import CoordUnit
from core.coordinates.exp_factor_coord import CoordTask, CoordAttention, CoordCategory
from core.coordinates.time_coord import CoordTime

from core.data_structures.base_data_structure import DataStructure
from core.attributes.brain_info import Area, Training


class FiringRatesPop(DataStructure[CoreRates]):
    """
    Firing rates for a pseudo-population in a set of pseudo-trials.

    Key Features
    ------------
    Dimensions : ``ensembles``, ``units``, ``folds``, ``trials``, ``time``

    Coordinates:

    - ``units`` (dimensions ``ensembles``, ``units``)
    - ``task``  (dimension ``trials``)
    - ``attention``   (dimension ``trials``)
    - ``category``  (dimension ``trials``)
    - ``time``  (dimension ``time``)

    Identity Metadata: ``area``, ``training`

    Descriptive Metadata: ``with_error``

    Attributes
    ----------
    area : Area
        Brain area from which the units were recorded.
    training : Training
        Training condition of the animals from which the units were recorded.
    with_error : bool
        Flag indicating whether the data set comprises error trials.
    data : CoreRates
        Firing rates time courses of all the units in all the trials.
        Shape: ``(n_ens, n_units, n_folds, n_trials, n_t)``, with:

        - ``n_ens``: Number of ensembles in the pseudo-population.
        - ``n_units``: Number of units in the pseudo-population.
        - ``n_folds``: Number of folds in the cross-validation scheme.
        - ``n_trials``: Number of pseudo-trials in the reconstructed data set.
        - ``n_t``: Number of time points in a trial's time course.

    units : CoordUnit
        Coordinate labels of the units in each ensemble of the pseudo-population.
        Dimensions: ``ensembles``, ``units``.
    task : CoordTask
        Coordinate labels for the task from which each trial comes.
    attn : CoordAttention
        Coordinate labels for the attentional state from which each trial comes.
    categ : CoordCategory
        Coordinate labels for the stimulus presented in each trial.
    time : CoordTime
        Time points of the firing rate time courses (in seconds).
    n_trials : int
        (Property) Number of trials in the subset.
    """

    # --- Data Structure Schema --------------------------------------------------------------------

    DIMENSIONS_SPEC = DimensionsSpec(
        ensembles=False,  # optional, if unique pseudo-population
        units=True,  # optional, if single unit
        folds=False,  # optional, if no cross-validation
        trials=True,
        time=False,  # optional, if time-averaged data
    )
    COMPONENTS_SPEC = ComponentSpec(
        data=CoreRates,
        units=CoordUnit,
        task=CoordTask,
        attention=CoordAttention,
        category=CoordCategory,
        time=CoordTime,
    )
    IDENTIFIERS = MappingProxyType(
        {
            "area": MetaDataField(Area, ""),
            "training": MetaDataField(Training, ""),
        }
    )

    def __init__(
        self,
        area: Area,
        training: Training,
        with_error: bool = False,
        data: CoreRates | None = None,
        **coords: Coordinate,
    ):
        # Set sub-class specific metadata
        self.area = area
        self.training = training
        self.with_error = with_error
        # Set data and coordinate attributes via the base class constructor
        super().__init__(data=data, **coords)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}>: Area {self.area}, Training {self.training}, "
            f"#trials={self.n_trials}" + super().__repr__()
        )

    # --- Getter Methods ---------------------------------------------------------------------------

    @property
    def n_trials(self) -> int:
        """Number of pseudo-trials (length of the dimensions `trials`)."""
        return self.get_data().get_size("trials")
