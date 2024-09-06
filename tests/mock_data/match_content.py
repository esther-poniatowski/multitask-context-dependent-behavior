#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mock_data.match_content` [module]

Fixtures to generate data corresponding to mock input files.

Directory of input files: `tests/mock_data/input_files`
"""
from pathlib import Path
from typing import Dict, List, Union

import pytest

# Relative path to mock data based on the file's location
PATH_MOCK_DATA = Path(__file__).parent / "input_files"
PATH_MOCK_ENV = PATH_MOCK_DATA / ".env"
PATH_MOCK_YAML = PATH_MOCK_DATA / "structure.yml"


StructureType = Dict[str, Union[Dict, str]]
"""Type alias for the directory structure."""


@pytest.fixture
def mock_credentials() -> Dict[str, Union[str, Path]]:
    """
    Provide the credentials corresponding to `.env`.

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
    Provide the python dictionary corresponding to `structure.yml`.

    Returns
    -------
    StructureType
        Nested dictionary.
    """
    return {"dir": {"subdir1": {}, "subdir2": {"subsubdir": {}}}}


@pytest.fixture
def expected_paths(tmp_path) -> List[Path]:
    """
    Provide the list of expected paths based on the directory structure in `structure.yml`.

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


data_dict_yml: Dict[str, str] = {"key1": "value1", "key2": "value2"}
"""Expected content from `data_dict.yml`."""

data_list_yml: List[str] = ["value1", "value2"]
"""Expected content from `data_list.yml`."""
