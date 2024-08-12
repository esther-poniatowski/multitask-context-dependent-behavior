#!/usr/bin/env bash

# =================================================================================================
# Script Name: test.sh
# Description: Run the tests suite with :mod:`pytest`.
#
# Variables
# ---------
# TESTS_DIR
# 	Directory containing the test files.
# =================================================================================================

pytest $(TESTS_DIR)

# TODO: Export test report.
