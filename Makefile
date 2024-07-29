
# =================================================================================================
# Main Makefile for the workspace ("lead" part on the local machine).
#
# Targets
# -------
# workspace-info (default)
#   Display metadata about the workspace. Note: This command does not require to activate the conda
#   environment so that it can be used to check basic inclusion of the environment variables.
# activate
#   Run the activation command (see variable `ACTIVATE`) and display checks to ensure the correct
#   environment is activated.
#
# Variables
# ---------
# ENV_FILES : str
# 	Paths to the ``.env`` files where essential environment variables are defined.
# ACTIVATE : str
# 	Command to activate the conda environment. Defined as a variable to be reused in other targets.
#   Actions:
#     - Initialize the conda shell via the `shell.bash` hook.
#     - Activate the conda environment.
#     - Change the directory to the root of the workspace.
#
# Warning
# -------
# This file should be located at the *root* of the workspace to ensure paths consistency. This
# makefile plays the role of the entry point for the workspace. It includes other more specific
# makefiles usually located at the root of several component directories.
#
# Each included makefile should define a `help-...` target to display the list of available targets.
# The name of this target should be unique to avoid conflicts across makefiles when including them.
#
# Notes
# -----
# `.DEFAULT_GOAL` (special variable): Set the default target. If not specified, the default target
# corresponds to the *first* target defined in the Makefile, which is one of the *included*
# makefiles.
#
# See Also
# --------
# `.env` file for the definition of the other environment variables:
#
# - ROOT
# - WORKSPACE
# - MAKEFILE_PATHS
# - ENV_NAME
#
# =================================================================================================

ENV_FILES := meta.env path.env

$(info -----------------)
$(info INCLUDE ENVIRONMENT VARIABLES)
$(foreach FILE, $(ENV_FILES), \
	$(if $(wildcard $(FILE)), \
		$(eval ENV_FILE := $(FILE)) $(info Include: $(FILE)), \
		$(warning No environment file at $(FILE)) \
	) \
)

$(info -----------------)
$(info INCLUDE MAKEFILES)
$(foreach FILE_PATH, $(MAKEFILE_PATHS), \
    $(if $(wildcard $(FILE_PATH)), \
        $(eval include $(FILE_PATH)) $(info Include: $(FILE_PATH)), \
        $(warning No Makefile at $(FILE_PATH)) \
    ) \
)
$(info -----------------)

.DEFAULT_GOAL := workspace-info

.PHONY: workspace-info
workspace-info:
	@echo "-- WORKSPACE INFO"
	@echo "-- Workspace        : ${WORKSPACE}"
	@echo "-- Root directory   : ${ROOT}"
	@echo "-- Sub-directories  :"
	@cd ${ROOT}; ls
	@echo "--------------------"


ACTIVATE := @eval "$$(conda shell.bash hook)" && conda activate ${ENV_NAME} && cd ${ROOT}

.PHONY: env-info
env-info:
	${ACTIVATE} && \
	echo "-- CONDA ENVIRONMENT INFO" && \
	echo "-- Conda Environment: ${ENV_NAME}" && \
	echo "-- Location         : $${CONDA_PREFIX}" && \
	echo "-- Python Version   : $$(python --version)" && \
	echo "-- Python Path      : $$(which python)" && \
	echo "--------------------"
