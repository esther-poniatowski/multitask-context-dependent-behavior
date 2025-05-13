#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pylint configuration loader

Dynamically set Pylint configuration based on the directory being linted.

Warning
-------
The directory containing this plugin file (:mod:`pylint_configurer.py`) must be in the `PYTHONPATH`
to be loaded by Pylint.

Arguments
---------
linter : PyLinter
    Pylint linter object to configure.
"""
import configparser
import os
import sys
from pylint.lint import PyLinter


def register(linter: PyLinter):
    """Register the plugin (automatically called by Pylint when it loads the plugin)."""
    load_config(linter)


CONF_BASE = 'pylintrc_base.ini'
CONF_MAP = {
    'tests': 'pylintrc_tests.ini',
    'data': 'pylintrc_disable.ini',
}


def load_config(linter: PyLinter):
    """
    Load the appropriate configuration file for the directory being linted.

    Implementation
    --------------
    1. Load the base configuration file by default.
    2. Get the files/directories being linted. They are retrieved from the command line arguments.
    3. Identify paths that need specific configurations based on the directory name.
    4. Load the specific configuration file for each identified path if needed.
    """
    # Load the base configuration
    base_conf_path = os.path.join(os.path.dirname(__file__), CONF_BASE)
    if os.path.exists(base_conf_path):
        apply_config(linter, base_conf_path)
    else:
        print(f"Base configuration file not found: {base_conf_path}")
    # Get files/directories being linted
    linted_paths = sys.argv[1:]
    # Load specific configurations if needed
    for path in linted_paths:
        for target_dir, conf_file in CONF_MAP.items():
            if target_dir in path:
                conf_path = os.path.join(os.path.dirname(__file__), conf_file)
                if os.path.exists(conf_path):
                    apply_config(linter, conf_path)
                else:
                    print(f"Specific configuration file not found: {conf_path}")
                break  # load specific config only once per path


def apply_config(linter: PyLinter, config_path: str):
    """
    Apply configurations from an INI file to the linter.

    Arguments
    ---------
    config_path : str
        Path to the configuration file to load.

    Implementation
    --------------
    Format of the configuration file: INI file with sections and options.

    `linter.config`: `pylint.config.Configuration` object, whose attributes correspond to the
    configuration options.
    """
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)

    for section in config.sections():
        for option, value in config.items(section):
            # Convert option names to match the linter's configuration attributes
            option = option.replace('-', '_')
            if hasattr(linter.config, option):
                # Handle multi-line values
                if value is not None:
                    value = value.replace('\n', '').strip()
                setattr(linter.config, option, value)
            else:
                print(f"Unknown configuration option: {option}")


def display_config_options(linter: PyLinter):
    """Display all configuration options available in the linter.config object."""
    config_options = vars(linter.config)
    for option in config_options:
        print(f"{option}: {getattr(linter.config, option)}")
