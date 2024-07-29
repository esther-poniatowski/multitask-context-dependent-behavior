# =================================================================================================
# Makefile for tests
#
# Targets
# -------
# test
# 	Run the full tests suite with :mod:`pytest`.
#
# Variables
# TESTS_DIR
# 	Directory containing the test files. Already exported as an environment variable.
# =================================================================================================

.PHONY: help-test
help-test:
	@echo "test 	 - Run the full tests suite with pytest."
	@echo ""
	@echo "TESTS_DIR = $(TESTS_DIR)"

.PHONY: test
test:
	pytest $(TESTS_DIR)

# TODO: Add a target for the test report.
