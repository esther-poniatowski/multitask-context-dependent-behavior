#!/usr/bin/env bash

# ============================================================================
# Script Name:   conda_env_mng.sh
# Description:   Manage a Conda environment.
#
# Functions
# ---------
# env_exists
# activate_env
# activate_base
# create_env
# update_env
# remove_env
# symlink_in_activate_d
# inspect_env
#
# Notes
# -----
# The single quotes in messages only enhace readability by distinguishing a variable from the
# surrounding text. The variable itself is still expanded because the whole string is enclosed in
# double quotes.
# ============================================================================


# Function:    env_exists
# Description: Check is a conda environment already exists with the given name.
#
# Arguments
# ---------
# env_name : str
#     Name of the Conda environment. Default : Value of the ENV_NAME variable.
#
# Returns
# -------
# exists : bool
#     True if the environment exists, False otherwise.
#
# Usage
# -----
# .. code-block:: bash
#
#    env_exists <ENV_NAME>
#
# Notes
# -----
# The output of the ``conda env list`` command is piped to ``grep`` to search for the environment
# name.
# ``-q`` option: Suppress output, only return the exit status.
#
env_exists() {
    local env_name="${1:-$ENV_NAME}"
    conda env list | grep -q "${env_name}" && {
        echo "[SUCCESS] Environment '${env_name}' exists."
        return 0
    } || {
        echo "[ERROR] Environment '${env_name}' does not exist yet."
        return 1
    }
}


# Function:    activate_env
# Description: Activate a Conda environment in the current shell.
#
# Arguments
# ---------
# env_name : str
#     Name of the Conda environment. Default : Value of the ENV_NAME variable.
#
# Raises
# ------
# [ERROR] If the environment activation fails.
#
# Usage
# -----
# .. code-block:: bash
#
#    activate_env <ENV_NAME>
#
# Notes
# -----
# In bash scripts, it is necessary to source the ``activate`` script of conda to activate the
# environment. Is not sufficient to use ``conda activate`` directly.
#
activate_env() {
    local env_name="${1:-$ENV_NAME}"
    source activate "${env_name}" || {
        echo "[ERROR] Failed to activate environment '${env_name}'."
        exit 1
    }
    echo "[SUCCESS] Environment '${env_name}' is active."
}


# Function:    activate_base
# Description: Activate the base Conda environment in the current shell.
#
# Raises
# ------
# [ERROR] If the environment deactivation fails.
#
# Usage
# -----
# .. code-block:: bash
#
#    activate_base
#
# Notes
# -----
# The base environment is required for some operations, such as creating or removing environments.
#
activate_base() {
    source activate base || {
        echo "[ERROR] Failed to activate 'base' environment."
        exit 1
    }
    echo "[SUCCESS] Base environment is active."
}


# Function:    create_env
# Description: Create a Conda environment from a YAML file.
#
# Arguments
# ---------
# environment_yaml : str
#     Path to the environment YAML file. Default : Value of the ENVIRONMENT_YAML variable.
#
# Raises
# ------
# [ERROR] If the environment creation fails.
#
# Usage
# -----
# .. code-block:: bash
#
#    create_env path/to/environment.yml
#
# Notes
# -----
# This operation requires the *base* environment to be active.
#
create_env() {
    local environment_yaml="${1:-$ENVIRONMENT_YAML}"
    activate_base
    conda env create --file "${environment_yaml}" && {
        echo "[SUCCESS] Created environment from '${environment_yaml}'."
    } || {
        echo "[ERROR] Failed to create environment from '${environment_yaml}'."
        exit 1
    }
}


# Function:    update_env
# Description: Update the environment using a YAML file.
#
# Raises
# ------
# [ERROR] If the environment update fails.
#
# Usage
# -----
# .. code-block:: bash
#
#    update_env path/to/environment.yml
#
# Notes
# -----
# Option ``--prune``: Remove dependencies that are no longer required.
#
update_env() {
    local environment_yaml="${1:-$ENVIRONMENT_YAML}"
    conda env update --file "${environment_yaml}" --prune && {
        echo "[SUCCESS] Updated environment to match '${environment_yaml}'."
    } || {
        echo "[ERROR] Failed to update environment to match '${environment_yaml}'."
        exit 1
    }
}


# Function:    remove_env
# Description: Remove a Conda environment.
#
# Arguments
# ---------
# env_name : str
#     Name of the Conda environment. Default : Value of the ENV_NAME variable.
#
# Raises
# ------
# [ERROR] If the environment removal fails.
#
# Usage
# -----
# .. code-block:: bash
#
#    remove_env <ENV_NAME>
#
# Notes
# -----
# ``-y`` option: Automatically confirm the removal.
# This operation requires the base environment to be active.
#
remove_env() {
    local env_name="${1:-$ENV_NAME}"
    activate_base
    conda env remove --name "${env_name}" -y && {
        echo "[SUCCESS] Removed environment '${env_name}'."
    } || {
        echo "[ERROR] Failed to remove environment '${env_name}'."
        exit 1
    }
}


# Function:    get_activate_d
# Description: Get the path to the 'activate.d' directory of the active Conda environment.
#
# Returns
# -------
# activate_d : str
#     Path to the 'activate.d' directory of the active Conda environment.
#
# Usage
# -----
# .. code-block:: bash
#
#    get_activate_d <ENV_NAME>
#
# Notes
# -----
# This operation requires the environment to be active so that the appropriate ``$CONDA_PREFIX``
# variable is set.
#
get_activate_d() {
    local activate_d="${CONDA_PREFIX}/etc/conda/activate.d"
    echo "${activate_d}"
}


# Function:    symlink_in_activate_d
# Description: Create a symlink in the 'activate.d' directory of the active Conda environment.
#
# Arguments
# ---------
# target_path : str
#     Path to the file or directory in the workspace to be symlinked in 'activate.d'.
# alias : str
#     Alias for the symlink in the 'activate.d' directory.
#
# Raises
# ------
# [ERROR] If the symlink creation fails.
#
# Usage
# -----
# To create a symlink to a path with an alias in 'activate.d' of the active environment:
#
# .. code-block:: bash
#
#    symlink_in_activate_d <target_path> <alias>
#
# Notes
# -----
# This operation requires the environment to be active so that the ``$CONDA_PREFIX`` variable is set.
#
# ``realpath`` : Resolve the absolute path of the target path, to ensure that the symlink points to
# the correct location regardless of the current working directory.
#
symlink_in_activate_d() {
    local target_path="${1}"
    local alias="${2}"
    local activate_d="$(get_activate_d)"
    local symlink_path="${activate_d}/${alias}"
    # Check if target path exists
    if [ ! -e "${target_path}" ]; then
        echo "[ERROR] Target path '${target_path}' does not exist."
        exit 1
    fi
    # Create the 'activate.d' directory if it does not exist
    mkdir -p "${activate_d}"
    # Remove existing symlink or file with the same name
    if [ -L "${symlink_path}" ] || [ -e "${symlink_path}" ]; then
        rm "${symlink_path}"
    fi
    # Create the new symlink
    ln -s "$(realpath "${target_path}")" "${symlink_path}" && {
        echo "[SUCCESS] Created symlink at '${symlink_path}' for '${target_path}'."
    } || {
        echo "[ERROR] Failed to create symlink at '${symlink_path}' for '${target_path}'."
        exit 1
    }
}


# Function:    inspect_env
# Description: Display information about the active Conda environment.
#
# Arguments
# ---------
# Usage
# -----
# .. code-block:: bash
#
#    inspect_env
#
# Notes
# -----
# This operation requires the environment to be active so that the environment variables are set.
#
inspect_env() {
    local activate_d="$(get_activate_d)"
    echo "----------------"
    echo "ENVIRONMENT INFO"
    echo "Conda Environment : ${CONDA_DEFAULT_ENV}"
    echo "Location          : ${CONDA_PREFIX}"
    echo "Python Version    : $(python --version)"
    echo "Python Interpreter: $(which python)"
    echo "Contents of the 'activate.d' directory (${activate_d}):"
    ls -l "${activate_d}" || {
        echo "[ERROR] Failed to list contents of 'activate.d' directory."
        exit 1
    }
    echo "--------------------"
}
