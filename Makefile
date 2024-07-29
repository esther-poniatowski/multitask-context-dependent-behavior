
# =================================================================================================
# Main Makefile for the project.
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
# ENV_FILE
# 	Path to the ``.env`` file where essential environment variables are stored.
# MAKEFILE_PATHS (from ``.env``)
# 	List of paths to the Makefiles to include.
# ENV_NAME (from ``.env``)
# 	Name of the conda environment to activate.
# ROOT (from ``.env``)
# 	Path to the root of the project.
# ACTIVATE (defined in the current Makefile)
# 	Command to activate the conda environment. Defined as a variable to be reused in other targets.
#   Actions:
#     - Initialize the conda shell via the `shell.bash` hook.
#     - Activate the conda environment.
#     - Change the directory to the root of the project.
#
# Warning
# -------
# This file should be located at the *root* of the project to ensure paths consistency. This
# makefile plays the role of the entry point for the project. It includes other more specific
# makefiles usually located at the root of several component directories.
#
# Notes
# -----
# `.DEFAULT_GOAL` is a special variable that sets the default target. If not specified, the default
# target corresponds to the first target defined in the Makefile. Here, because other makefiles are
# included, before defining any target, the default target would be set to one of the included ones.
# The `.DEFAULT_GOAL` variable is used to ensure that the `workspace-info` target is the default one.
#
# Each makefile included should define a `help-...` target to display the list of available targets.
# The name of this target should be unique to avoid conflicts acroos makefiles when including them.
#
# See Also
# --------
# =================================================================================================

ENV_FILE := .env

$(info -----------------)
$(info INCLUDE ENVIRONMENT VARIABLES)
ifneq ("$(wildcard $(ENV_FILE))","")
    include $(ENV_FILE) $(info Include: $(ENV_FILE))
endif

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
