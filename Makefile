
# =================================================================================================
# Main Makefile for the project.
#
# Targets
# -------
# test
# 	Run the tests.
# docs
# 	Build the documentation.
# open-docs
# 	Open the documentation in the browser.
#
# Variables
# ---------
# ROOT
# 	Root directory of the project.
# CONFIG_DIR
# 	Directory containing the configuration files.
#
# Warning
# -------
# This file should be located at the *root* of the project for the paths to be correct.
#
# See Also
# --------
# config/environment/Makefile
# config/deploy/Makefile
# =================================================================================================

# Set paths
ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CONFIG_DIR := $(ROOT)/config

.PHONY: all
all: help

.PHONY: help
help:
	@echo "Main Makefile for the project."
	@echo "Root directory: $(ROOT)"
	@echo "Configuration directory: $(CONFIG_DIR)"
	@echo "Include:"
	@echo "  - $(CONFIG_DIR)/environment/Makefile"
	@echo "  - $(CONFIG_DIR)/deploy/Makefile"

# Inlcude other Makefiles
include $(CONFIG_DIR)/environment/Makefile
include $(CONFIG_DIR)/deploy/Makefile


.PHONY: test
test:
	pytest

.PHONY: docs
docs:
	sphinx-build -b html docs/source/ docs/build/html
	@open docs/build/html/index.html

.PHONY: open-docs
open-docs:
	@open docs/build/html/index.html
