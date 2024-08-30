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
	# Get the path to the reports directory in the workspace (environment variable)
	reports_dir = os.environ.get('REPORTS_DIR', 'reports')
	# Ensure the reports directory exists
	os.makedirs(reports_dir, exist_ok=True)
	if config.option.xmlpath is None:
		config.option.xmlpath = ""
