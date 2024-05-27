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
sys.path.insert(0, os.path.abspath('../tests'))


# -- Project information -----------------------------------------------------

project = 'Multi-Task Context-Dependent Behavior'
copyright = '2025, Esther Poniatowski'
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
    'sphinx_rtd_theme',
]

exclude_patterns = [
	'.DS_Store', '*~', '.vscode',
	'__pycache__', '*.pyc', '*.pyo', 
	'.pytest_cache', '_build']


# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
#html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"


# -- Autodoc Settings --------------------------------------------------------

# Display settings for API members
autodoc_default_options = {
    'member-order': 'bysource', # ordering according to source code
    'special-members': '__init__', # include __init__ method
    'undoc-members': True, # include members without documentation
    'exclude-members': '__weakref__'
}
# Document type hints
autodoc_typehints = "description" 
add_module_names = False


# -- Napoleon Settings --------------------------------------------------------

# Custom types for napoleon
napoleon_use_param = True
napoleon_type_aliases = {"ArrayLike": ":obj:`mtcdb.types.ArrayLike`",
                         "NumpyArray": ":obj:`mtcdb.types.NumpyArray`",
}
# Custom sections for napoleon
napoleon_custom_sections = [('Test Inputs', 'params_style'), # for test docs
                            ('Expected Outputs', 'params_style'), # for test docs
                            ('Sub-Packages', 'params_style'), # for package docs
                            ('Modules', 'params_style'), # for module docs
                            ('Files', 'params_style'), # for module docs
                            'Algorithm', 
                            'Implementation'
                            ] 


# -- Intersphinx Settings ----------------------------------------------------

# External links to python libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pytest': ('https://docs.pytest.org/en/stable/', None),
}


# -- Test Reports Settings ---------------------------------------------------

# Location of the template for the test report
tr_report_template = '_templates/test_report_template.txt'
