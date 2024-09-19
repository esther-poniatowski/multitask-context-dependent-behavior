#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_utils.test_path_system.test_storage_rulers` [module]

See Also
--------
:mod:`utils.storage_rulers.base_path_ruler`: Tested module.
:mod:`utils.storage_rulers.impl_path_rulers`: Tested module.
"""
import os

import pytest

from utils.storage_rulers.base_path_ruler import PathRuler
from utils.storage_rulers.impl_path_rulers import RawSpkTimesPath


def test_get_root(tmp_path):
    """
    Test :meth:`PathRuler.get_root` for the default value of the data root directory.

    Test Inputs
    -----------
    tmp_path : Path
        Temporary directory created by pytest fixture. It is exported as an environment variable
        under the name "DATA_DIR" in the current shell.

    Expected Output
    ---------------
    Value of the environment variable "DATA_DIR".
    """
    data_dir = tmp_path
    os.environ["DATA_DIR"] = str(data_dir)
    assert PathRuler.get_root() == data_dir, "Root directory not retrieved"


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
    unit = "avo052a-d1"
    session = "avo052a04_p_PTD"
    path_manager = RawSpkTimesPath(tmp_path)
    expected_path = tmp_path / "raw" / unit / session
    assert path_manager.get_path(unit, session) == expected_path, "Incorrect constructed path."


@pytest.mark.skip(reason="Not implemented")
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
    area = "A1"
    training = "Trained"
    path_manager = None  # TODO: FiringRatesPath(tmp_path)
    expected_path = tmp_path / "processed" / "populations" / f"{area}_{training}"
    assert path_manager.get_path(area, training) == expected_path, "Incorrect constructed path."
