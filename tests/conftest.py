#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conftest
========

Configuration file for pytest.

Notes
-----
- Set the path for exporting test results to XML, to be used by Sphinx.
"""

import sys
import pytest

def pytest_configure(config):
	if config.option.xmlpath is None:
		config.option.xmlpath = "docs/source/code/tests/pytest_results.xml"

