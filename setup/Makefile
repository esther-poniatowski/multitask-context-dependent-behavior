# =================================================================================================
# Makefile to manage the virtual environment of the project.
#
# Targets
# -------
# help-setup
#     Display the list of available targets.
# create-env
#     Create the virtual environment from the environment file (`ENVIRONMENT_YAML`).
# update-env
#     Update the virtual environment from the environment file (`ENVIRONMENT_YAML`).
#     Option `--prune`: Remove the packages not listed in the environment file.
# set-vars
#    Set the environment variables defined in `ENV_FILE`.
# check-vars
#    Display the environment variables.
# set-pkg
#    Register the packages defined in `DEV_PKG` in development mode.
# check-pkg
#    Display the packages registered in development mode.
# setup-workspace
#    Setup the full workspace:
# 	 1. Create the conda virtual environment
#	 2. Set the environment variables
#	 3. Register the packages in development mode.
# update-workspace
#    Update the full workspace.
# check-workspace
#    Display metadata about the workspace to ensure setup success.
# remove-env
#    Remove the virtual environment and clean the project from cache files.
#
# Variables
# ---------
# ENV_NAME : str
#     Name of the conda virtual environment.
# ENVIRONMENT_YAML : str
#     Path to the environment file.
# ENV_FILE : str
#     Path to the ``.env`` file where essential environment variables are stored.
# SET_ENV_VARS : str
#     Path to the python script which sets the environment variables.
# DEV_PKG : str
#     List of packages to register in development mode.
# MANAGE_PKG : str
#     Path to the python script which manages the packages via a class (methods: clear, register, check).
#
# Notes
# -----
# In ``create-env`` and ``update-env`` targets, the name of the environment is *not required* since
# it is defined in the environment file. Similarly, it is not required to activate the environment
# to update it.
#
# Warning
# -------
# The setup command should be invoked from the *root directory* of the project, only once.
# =================================================================================================


.PHONY: help-setup
help-setup:
	@echo "create-env       - Create the virtual environment from the environment file."
	@echo "update-env       - Update the virtual environment from the environment file."
	@echo "set-vars         - Set the environment variables defined in the `.env` file."
	@echo "check-vars       - Display the environment variables."
	@echo "set-pkg          - Register the packages defined in `DEV_PKG` in development mode."
	@echo "check-pkg        - Display the packages registered in development mode."
	@echo "setup-workspace  - Setup the full workspace."
	@echo "update-workspace - Update the full workspace."
	@echo "check-workspace  - Display metadata about the workspace."
	@echo "remove-env       - Remove the virtual environment and clean the project from cache files."


.PHONY: create-env
create-env:
	conda env create --file ${ENVIRONMENT_YAML}

.PHONY: update-env
update-env:
	conda env update --file ${ENVIRONMENT_YAML} --prune

.PHONY: set-vars
set-vars:
	${ACTIVATE} && \
	python ${SET_ENV_VARS} --env-path ${ENV_FILE}

.PHONY: check-vars
check-vars:
	${ACTIVATE} && python ${SET_ENV_VARS} --method check

.PHONY: set-pkg
set-pkg:
	${ACTIVATE} && \
	python ${MANAGE_PKG} --method clear && \
	python ${MANAGE_PKG} --method register --path-pkg ${DEV_PKG}

.PHONY: check-pkg
check-pkg:
	${ACTIVATE} && \
	python ${MANAGE_PKG} --method check

.PHONY: check-workspace
check-workspace:
	${ACTIVATE} && \
	echo "Conda Environment : ${CONDA_DEFAULT_ENV}" && \
	echo "Location          : ${CONDA_PREFIX}" && \
	python ${MANAGE_PKG} --method check

.PHONY: setup-workspace
setup-workspace: create-env set-vars set-pkg

.PHONY: update-workspace
update-workspace: update-env set-vars set-pkg

.PHONY: remove-env
remove-env:
	conda env remove --name ${ENV_NAME} -y
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -exec rm -f {} +
