#!/usr/bin/env bash

# ==================================================================================================
# Script Name:   init.sh
# Description:   Initialize a Conda environment for the workspace.
#
# Usage
# -----
# Run this script from the root directory of the workspace.
#
# .. code-block:: bash
#
#   bash setup/init.sh
#
# Warning
# -------
# This file should be executed from the *root* of the workspace to ensure paths consistency.
#
# See Also
# --------
# path.env
#   Define variables: CONDA_ENV_MNG, ENV_NAME, ENVIRONMENT_YAML, POST_ACTIVATE_SCRIPT
# src/tasks/environment/conda_env_mng.sh
#   Define functions: env_exists, update_env, create_env, activate_env, symlink_in_activate_d, inspect_env
# ==================================================================================================

echo "==========================================="
echo "Initialization Script - Environment 'mtcdb'"
echo "==========================================="

# --- Imports --------------------------------------------------------------------------------------

# Root directory of the workspace (current directory)
ROOT="$(pwd)"

# Set Environment Variables
ENV_FILE="${ROOT}/path.env"  # path to the path.env file *relative to root*
if [ -f "${ENV_FILE}" ]; then
    set -a
    source "${ENV_FILE}"
    set +a
    echo "[SUCCESS] Set environment variables from '${ENV_FILE}'"
else
    echo "[ERROR] File not found at '${ENV_FILE}'."
    exit 1
fi

# Import Utils
if [ -f "${CONDA_ENV_MNG}" ]; then
    source "${CONDA_ENV_MNG}"
else
    echo "[ERROR] Utils not found at ${CONDA_ENV_MNG}"
    exit 1
fi

# --- Main Process ---------------------------------------------------------------------------------

# Check if the environment already exists
if env_exists "${ENV_NAME}"; then
    # Update the existing Conda environment to match the environment.yml file
    echo "Update Conda environment '${ENV_NAME}'"
    update_env "${ENVIRONMENT_YAML}"
else
    # Create a new conda environment from the environment.yml file
    echo "Create Conda environment '${ENV_NAME}'"
    create_env "${ENVIRONMENT_YAML}"
fi

# Activate the Conda environment
echo "Activate Conda environment '${ENV_NAME}'"
activate_env "${ENV_NAME}"

# Create symlinks to the local post-activation script and root directory
echo "Symlink post_activate.sh in 'activate.d'"
symlink_in_activate_d "${POST_ACTIVATE_SCRIPT}" "post_activate.sh"

echo "Symlink root directory in 'activate.d'"
symlink_in_activate_d "${ROOT}" "root"

# Inspect the environment
inspect_env

echo " === END ==="
