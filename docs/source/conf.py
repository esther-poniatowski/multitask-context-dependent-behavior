#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conf
====

Configuration settings for the Sphinx documentation builder.
"""

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add source code directory to locations accessible for import
sys.path.insert(0, os.path.abspath('../src')) 


# -- Project information -----------------------------------------------------

project = 'Multi-Task Context-Dependent Behavior'
copyright = '202, Esther Poniatowski'
author = 'Esther Poniatowski'
release = '0.1.0'
version = '0.1.0'
language = 'en'


# -- General Configuration ---------------------------------------------------


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
    'sphinx_needs',
    'sphinxcontrib.test_reports',
]

exclude_patterns = [
	'.DS_Store', '*~', '.vscode',
	'__pycache__', '*.pyc', '*.pyo', 
	'.pytest_cache', '_build']


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Extensions Settings -----------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = "description" # document type hints

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

tr_report_template = '_templates/test_report_template.txt'