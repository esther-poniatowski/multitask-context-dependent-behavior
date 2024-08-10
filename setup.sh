#!/usr/bin/env bash

# ============================================================================
# Script Name:   setup.sh
# Description:   Manage a Conda environment using a YAML file
#                Automatically executed upon user input
#
# Functionalities:
#
# - Create a new Conda environment from a YAML file.
# - Update the existing Conda environment using the YAML file.
# - Remove the Conda environment.
# - Symlink or update the post-activate script in the environment.
# - Display information about the current Conda environment.
#
# Usage
# -----
# .. code-block:: bash
#
#   ./setup.sh {create|update|remove|link|inspect}
#
# Options
# -------
# create
#   Create a new Conda environment using the YAML file.
# update
#   Update the existing Conda environment using the YAML file.
# remove
#   Remove the Conda environment.
# link
#   Create or update a symlink to the post-activate script in the environment directory.
# inspect
#   Display information about the current Conda environment.
#
# Variables
# ---------
# ENV_NAME (str)
#   Name of the Conda environment.
# ENVIRONMENT_YAML
#   Path to the environment YAML file.
# POST_ACTIVATE_SCRIPT
#   Initial path to the post-activate script.
#
# Notes
# -----
# The single quotes in messages only enhace readability by distinguishing a variable from the
# surrounding text. The variable itself is still expanded because the whole string is enclosed in
# double quotes.
# ============================================================================

# === Variables ==============================================================

ENV_NAME="mtcdb"
ENVIRONMENT_YAML="config/environment.yml"
POST_ACTIVATE_SCRIPT="config/post_activate.sh"


# === Functions ==============================================================

# Function:    ensure_env_active
# Description: Ensure the correct environment is active.
#
# Raises
# ------
# Error: If the environment is not activated.
#
# Example
# -------
# .. code-block:: bash
#
#    ensure_env_active
#
ensure_env_active() {
    if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
        source activate "${ENV_NAME}" || {
            echo "Error: Failed to activate environment '${ENV_NAME}'."
            exit 1
        }
    fi
    echo "Success: Environment '${ENV_NAME}' is active."
}


# Function:    ensure_base_active
# Description: Ensure the base environment is active.
#
# Raises
# ------
# Error: If the base environment cannot be activated.
#
# Example
# -------
# .. code-block:: bash
#
#    ensure_base_active
#
ensure_base_active() {
    if [[ "$CONDA_DEFAULT_ENV" != "base" ]]; then
        conda deactivate || {
            echo "Error: Failed to deactivate environment."
            exit 1
        }
    fi
    echo "Success: Base environment is active."
}


# Function:    create_env
# Description: Create a Conda environment from a file.
#
# Raises
# ------
# Error: If the environment creation fails.
#
# Example
# -------
# .. code-block:: bash
#
#    create_env
#
create_env() {
    ensure_base_active
    conda env create --file "${ENVIRONMENT_YAML}" && {
        echo "Success: Environment '${ENV_NAME}' created."
    } || {
        echo "Error: Failed to create environment '${ENV_NAME}'."
        exit 1
    }
    link_post_activate_script
}


# Function:    update_env
# Description: Update the environment using the YAML file.
#
# Notes
# -----
# - The --prune option removes dependencies that are no longer required.
#
# Raises
# ------
# Error: If the environment update fails.
#
# Example
# -------
# .. code-block:: bash
#
#    update_env
#
update_env() {
    ensure_env_active
    conda env update --file "${ENVIRONMENT_YAML}" --prune && {
        echo "Success: Environment '${ENV_NAME}' updated."
    } || {
        echo "Error: Failed to update environment '${ENV_NAME}'."
        exit 1
    }
    link_post_activate_script
}


# Function:    remove_env
# Description: Remove the Conda environment.
#
# Notes
# -----
# - The -y option automatically confirms the removal.
#
# Raises
# ------
# Error: If the environment removal fails.
#
# Example
# -------
# .. code-block:: bash
#
#    remove_env
#
remove_env() {
    ensure_base_active
    conda env remove --name "${ENV_NAME}" -y && {
        echo "Success: Environment '${ENV_NAME}' removed."
    } || {
        echo "Error: Failed to remove environment '${ENV_NAME}'."
        exit 1
    }
}


# Function:    link_post_activate_script
# Description: Symlink to the post-activate script.
#
# Implementation
# --------------
# - ``[ -L "${SYMLINK_PATH}" ]`` : Check if a symbolic link exists at the specified path.
# - ``[ -e "${SYMLINK_PATH}" ]`` : Check if a file (or symlink) exists at the specified path.
# - ``realpath "${POST_ACTIVATE_SCRIPT}"`` : Resolve the absolute path of the POST_ACTIVATE_SCRIPT,
# to ensure that the symlink points to the script regardless of the current working directory.
#
# Raises
# ------
# Error: If the symlink creation fails.
#
# Example
# -------
# .. code-block:: bash
#
#    link_post_activate_script
#
link_post_activate_script() {
    ensure_env_active
    ACTIVATE_D_DIR="${CONDA_PREFIX}/etc/conda/activate.d" # Target directory
    SYMLINK_PATH="${ACTIVATE_D_DIR}/post_activate.sh"     # Symlink path
    mkdir -p "${ACTIVATE_D_DIR}"
    # Remove existing symlink or file if it exists
    if [ -L "${SYMLINK_PATH}" ] || [ -e "${SYMLINK_PATH}" ]; then
        rm "${SYMLINK_PATH}"
    fi
    # Create the new symlink
    ln -s "$(realpath "${POST_ACTIVATE_SCRIPT}")" "${SYMLINK_PATH}" && {
        echo "Success: Symlink created at '${SYMLINK_PATH}'."
    } || {
        echo "Error: Failed to create symlink at '${SYMLINK_PATH}'."
        exit 1
    }
}

# Function:    inspect_env
# Description: Display information about the current Conda environment.
#
# Example
# -------
# .. code-block:: bash
#
#    inspect_env
#
inspect_env() {
    ensure_env_active
    ACTIVATE_D_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
    echo "----------------"
    echo "ENVIRONMENT INFO"
    echo "Conda Environment : ${ENV_NAME}, conda: ${CONDA_DEFAULT_ENV}"
    echo "Location          : ${CONDA_PREFIX}"
    echo "Python Version    : $(python --version)"
    echo "Python Interpreter: $(which python)"
    echo "Contents of the 'activate.d' directory (${ACTIVATE_D_DIR}):"
    ls -l "${ACTIVATE_D_DIR}" || {
        echo "Error: Failed to list contents of 'activate.d' directory."
        exit 1
    }
    echo "--------------------"
}


# === Main Script ============================================================

# Parse user options and execute the corresponding function
case $1 in
    create) create_env ;;
    update) update_env ;;
    remove) remove_env ;;
    link) link_post_activate_script ;;
    inspect) inspect_env ;;
    *) echo "Usage: $0 {create|update|remove|link|inspect}" ;;
esac
