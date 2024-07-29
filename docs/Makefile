# =================================================================================================
# Makefile for Documentation with Sphinx
#
# Targets
# -------
# docs
# 		Build the documentation in HTML format.
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

.PHONY: help-docs
help-docs:
	@echo "docs       - Build the documentation in HTML format."
	@echo "open-docs  - Open the documentation in the browser."
	@echo ""
	@echo "DOCS_SOURCE = $(DOCS_SOURCE)"
	@echo "DOCS_BUILD  = $(DOCS_BUILD)"
	@echo "DOCS_ENTRY  = $(DOCS_ENTRY)"

.PHONY: docs
docs:
	sphinx-build -b html $(DOCS_SOURCE) $(DOCS_BUILD)

.PHONY: open-docs
open-docs:
	@open $(DOCS_ENTRY)
