#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.storage_rulers.impl_path_rulers` [module]

Implementations of path generation rules.

Classes
-------
:class:`SpikeTimesRawPath`
:class:`SpikeTrainsPath`
:class:`FiringRatesUnitPath`
:class:`FiringRatesPopPath`
:class:`DecoderPath`

See Also
--------
:class:`utils.storage_rulers.base_path_ruler.PathRuler` : Interface to manage file paths.

Notes
-----
Disable the pylint warning ``arguments-differ``.
This warning is raised because the signature of the concrete ``get_path`` methods is different from
the one in the base class. This is intentional since the arguments depend on the specific data
structure associated to each path manager.
"""
# pylint: disable=arguments-differ

from pathlib import Path
from typing import Union

from utils.storage_rulers.base_path_ruler import PathRuler


# --- Raw Data -------------------------------------------------------------------------------------


class EventsPropertiesPath(PathRuler):
    """Path generation rules used by `EventsProperties` data structures."""

    def get_path(self, session: str) -> Path:
        """
        Construct the path for the metadata of one session.

        Parameters
        ----------
        session: str

        Returns
        -------
        Path
            Format: ``{root}/raw/expt_events/{session}``
        """
        return self.root_data / "raw" / "expt_events" / session


class SpikeTimesRawPath(PathRuler):
    """Path generation rules used by `SpikeTimesRaw` data structures."""

    def get_path(self, unit: str, session: str) -> Path:
        """
        Construct the path for the raw data of one unit in one session.

        Parameters
        ----------
        unit: str
        session: str

        Returns
        -------
        Path
            Format: ``{root}/raw/spikes/{unit}/{session}``
        """
        return self.root_data / "raw" / unit / session


# --- Preprocessed Data ----------------------------------------------------------------------------


class TrialsPropertiesPath(PathRuler):
    """Path generation rules used by `TrialsProperties` data structures."""

    def get_path(self, session: str) -> Path:
        """
        Construct the path for the properties of the trials recorded in one session.

        Parameters
        ----------
        session: str

        Returns
        -------
        Path
            Format: ``{root}/processed/sessions_info/{session}``
        """
        return self.root_data / "processed" / "sessions_info" / session


class SpikeTrainsPath(PathRuler):
    """Path generation rules used by `SpikeTrains` data structures."""

    def get_path(self, unit: str) -> Path:
        """
        Construct the path for a file storing the spike trains of one unit.

        Parameters
        ----------
        unit: str

        Returns
        -------
        Path
            Format: ``{root}/processed/units/{unit}_spk``
        """
        return self.root_data / "processed" / "spike_trains" / f"{unit}_spk"


class FiringRatesUnitPath(PathRuler):
    """Path generation rules used by `FiringRatesUnit` data structures."""

    def get_path(self, unit: str) -> Path:
        """
        Construct the path for a file storing firing rates.

        Parameters
        ----------
        unit: str

        Returns
        -------
        Path
            Format: ``{root}/processed/units/{unit}_fr``
        """
        return self.root_data / "processed" / "units" / f"{unit}_fr"


class FiringRatesPopPath(PathRuler):
    """Path generation rules used by :class:`FiringRatesPop` data structures."""

    def get_path(self, area: str, training: Union[str, bool]) -> Path:
        """
        Construct the path for a file storing firing rates.

        Parameters
        ----------
        area: str
        training: str
            Training status of the animal, usually 'Trained' or 'Naive'.
            If boolean, it is converted to a string.

        Returns
        -------
        Path
            Format: ``{root}/processed/populations/{area}_{training}``
        """
        return self.root_data / "processed" / "populations" / f"{area}_{training}"


# --- Analyses and Models --------------------------------------------------------------------------


class DecoderPath(PathRuler):
    """Path generation rules used by Decoder data structures."""

    def get_path(self, area: str, training: Union[str, bool], model: str) -> Path:
        """
        Construct the path for a file storing decoder weights.

        Parameters
        ----------
        area: str
        training: str or bool
            Training status of the animal, usually 'Trained' or 'Naive'.
            If bool, it is converted to a string.
        model: str
            Name of the decoder model.

        Returns
        -------
        Path
            Format: ``{root}/models/decoders/{model}/{area}_{training}``
        """
        if isinstance(training, bool):
            training = "Trained" if training else "Naive"
        return self.root_data / "models" / "decoders" / model / f"{area}_{training}"
