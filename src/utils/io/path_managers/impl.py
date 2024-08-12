#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io.path_managers` [module]

Implementations of path generation rules.

Classes
-------
:class:`RawSpkTimesPath`
:class:`FiringRatesPath`
:class:`DecoderPath`

See Also
--------
:class:`utils.io.path_manager_base.PathManager` : Interface to manage file paths.
"""

from pathlib import Path
from typing import Union

from utils.io.path_managers.base import PathManager


class RawSpkTimesPath(PathManager):
    """Path generation rules used by SpkTimes data structures."""

    def get_path(self, unit: str, session: str) -> Path:  # pylint: disable=arguments-differ
        """
        Construct the path for the raw data of one unit in one session.

        Parameters
        ----------
        unit: str
        session: str

        Returns
        -------
        Path
            Format: ``{root}/raw/{unit}/{session}``
        """
        return self.path_root / "raw" / unit / session


class SpikesTrainsPath(PathManager):
    """Path generation rules used by SpikesTrains data structures."""

    def get_path(self, unit: str) -> Path:  # pylint: disable=arguments-differ
        """
        Construct the path for a file storing the spike trains of one unit.

        Parameters
        ----------
        unit: str

        Returns
        -------
        Path
            Format: ``{root}/processed/units/{unit}``
        """
        return self.path_root / "processed" / "units" / unit


class FiringRatesPath(PathManager):
    """Path generation rules used by FiringRates data structures."""

    def get_path(
        self, area: str, training: Union[str, bool]
    ) -> Path:  # pylint: disable=arguments-differ
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
        return self.path_root / "processed" / "populations" / f"{area}_{training}"


class DecoderPath(PathManager):
    """Path generation rules used by Decoder data structures."""

    def get_path(
        self, area: str, training: Union[str, bool], model: str
    ) -> Path:  # pylint: disable=arguments-differ
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
        return self.path_root / "models" / "decoders" / model / f"{area}_{training}"
