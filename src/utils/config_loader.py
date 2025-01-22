#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`utils.config_loader` [module]

Load configuration files in YAML format.

Functions
---------
load_config

Usage
-----
Those utilities are used in the context of pipelines to recover default parameters from
configuration files.
"""
from pathlib import Path

import yaml


def load_config(config_path: str | Path, extensions=("yml, yaml")) -> dict:
    """
    Load YAML configuration file.

    Arguments
    ---------
    config_path : str
        Path to the configuration file.
    extensions : Tuple[str], default=("yml", "yaml")
        Supported extensions for YAML configuration files.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    ValueError
        If the configuration file is not in YAML format.

    Notes
    -----
    Structure of the configuration file:
    ```
    key1: value1
    key2: value2
    key3:
      key4: value4
      key5: value5
    ```

    Structure of the configuration dictionary:
    ```
    {
        "key1": "value1",
        "key2": "value2",
        "key3": {
            "key4": "value4",
            "key5": "value5"
        }
    }
    ```

    See Also
    --------
    yaml.safe_load : Load YAML configuration file.
    """
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix not in extensions:
            raise ValueError("Unsupported config format. Use YAML.")
        return yaml.safe_load(f)
