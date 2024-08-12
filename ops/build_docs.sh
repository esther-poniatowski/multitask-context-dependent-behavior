#!/usr/bin/env bash

# =================================================================================================
# Script Name: build_docs.sh
# Description: Build the documentation with Sphinx.
#
# Variables
# ---------
# DOCS_SOURCE
# 		Source directory for the documentation, where the input files are stored (rst, images...).
# DOCS_BUILD
# 		Build directory for the documentation in HTML format, where the output is stored.
# DOCS_ENTRY
# 		Entry point for the documentation, i.e. main HTML file (``index.html``).
# =================================================================================================

sphinx-build -b html $(DOCS_SOURCE) $(DOCS_BUILD)
