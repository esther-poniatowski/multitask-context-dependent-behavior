#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`conftest` [module]

Configuration file for pytest.

Notes
-----
Set the path for exporting test results to XML.
The report is used by Sphinx for the documentation.
"""
import os

import pytest

REPORT_NAME = 'test_results'
REPORT_EXTENSION = '.xml'

def pytest_configure(config):
	# Get the root directory of the project
	root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Set the reports directory path
	reports_dir = os.path.join(root_dir, 'reports')
	# Ensure the reports directory exists
	os.makedirs(reports_dir, exist_ok=True)
	if config.option.xmlpath is None:
		config.option.xmlpath = ""

test_black = [1,2,
			  3,4]
