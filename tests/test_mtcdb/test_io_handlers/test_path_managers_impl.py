#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_utils.test_path_managers_impl` [module]

See Also
--------
:mod:`mtcdb.io_handlers.path_managers.impl`: Tested module.
"""

import pytest

from mtcdb.io_handlers.path_managers.impl import RawSpkTimesPath, FiringRatesPath


def test_rawspktimes_path(tmp_path):
    """
    Test :class:`RawSpkTimesPath` for constructing the correct path.

    Test Inputs
    -----------
    tmp_path : Path 
        Root directory for data, created by pytest fixture.
    unit: str
    session: str

    Expected Output
    ---------------
    Path under the format : `tmp_path / "raw" / unit / session
    """
    unit = 'avo052a-d1'
    session = 'avo052a04_p_PTD'
    path_manager = RawSpkTimesPath(tmp_path)
    expected_path = tmp_path / "raw" / unit / session
    assert path_manager.get_path(unit, session) == expected_path, "Incorrect constructed path."


def test_firingrates_path(tmp_path):
    """
    Test :class:`FiringRatesPath` for constructing the correct path.

    Test Inputs
    -----------
    tmp_path : Path
        Root directory for data, created by pytest fixture.
    area: str
    training: str

    Expected Output
    ---------------
    Path under the format : `tmp_path / "processed" / "populations" / {area}_{training}`
    """
    area = 'A1'
    training = 'Trained'
    path_manager = FiringRatesPath(tmp_path)
    expected_path = tmp_path / "processed" / "populations" / f"{area}_{training}"
    assert path_manager.get_path(area, training) == expected_path, "Incorrect constructed path."
