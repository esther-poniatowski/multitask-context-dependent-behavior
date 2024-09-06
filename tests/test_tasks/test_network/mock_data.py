#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_utils.mock_data` [module]

Implementation
--------------
Mock data corresponding to mock file in directory: `tests/test_tasks/test_network/mock_data`.
"""
from pathlib import Path
from typing import Dict, List, Union

import pytest

# Relative path to mock data based on the file's location
PATH_MOCK_DATA = Path(__file__).parent / "mock_data"
PATH_MOCK_ENV = PATH_MOCK_DATA / ".env"
PATH_MOCK_YAML = PATH_MOCK_DATA / "structure.yml"


StructureType = Dict[str, Union[Dict, str]]
"""Type alias for the directory structure."""


@pytest.fixture
def mock_credentials() -> Dict[str, Union[str, Path]]:
    """
    Fixture - Provide the credentials corresponding to the mock `.env` file.

    Returns
    -------
    Dict[str, Union[str, Path]]
        Dictionary containing the user, host, and root path for the remote server.
    """
    return {
        "user": "test_user",
        "host": "111.111.1.1",
        "root_path": Path("/remote/root"),
    }


@pytest.fixture
def mock_structure() -> StructureType:
    """
    Fixture - Provide the python dictionary corresponding to the mock `.yml` file.

    Returns
    -------
    StructureType
        Nested dictionary.
    """
    return {"dir": {"subdir1": {}, "subdir2": {"subsubdir": {}}}}


@pytest.fixture
def expected_paths(tmp_path) -> List[Path]:
    """
    Fixture - Provide the list of expected paths based on the directory structure in `mock_structure`.

    Returns
    -------
    List[pathlib.Path]
        List of pathlib.Path objects.
    """
    paths = [
        tmp_path / "dir",
        tmp_path / "dir/subdir1",
        tmp_path / "dir/subdir2",
        tmp_path / "dir/subdir2/subsubdir",
    ]
    return paths
