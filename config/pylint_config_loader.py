#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pylint.config

conf_base = 'pylintrc_base.ini'
conf_mapping = {
  'tests': 'pylintrc_tests.ini',
  'data': 'pylintrc_disable.ini',
}

def load_config():
    # Load the base configuration
    conf_path = os.path.join(os.path.dirname(__file__), conf_base)
    if os.path.exists(conf_path):
        pylint.config.load_config_file(conf_path)
    else:
        print(f"Configuration file not found: {conf_path}")
    # Get files/directories being linted
    linted_paths = sys.argv[1:]
    # Load specific configurations if needed
    for path in linted_paths:
        for target_dir, conf_file in conf_mapping.items():
            if target_dir in path:
                conf_path = os.path.join(os.path.dirname(__file__), conf_file)
                if os.path.exists(conf_path):
                    pylint.config.load_config_file(conf_path)
                break  # load specific config only once per path

load_config()
