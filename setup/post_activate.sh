#!/usr/bin/env bash

# ==================================================================================================
# Script Name:   post_activate.sh
# Description:   Post-Activation Script for Conda Environment
#
# Usage
# -----
# The script is automatically executed upon activation of the Conda environment.
#
# Warning
# -------
# This file should be symlinked in the Conda environment's `activate.d` directory.
#
# See Also
# --------
# path.env
#   Define variables: ENV_VAR_MNG, NAVIGATE_UTILS, PTH_PYTHON, PTH_BIN
# src/tasks/environment/env_var_mng.sh
#   Define functions: add_from_file, show_paths
# src/utils/direcotiries/navigate.sh
#   Define functions: navigate_to_dir
# ==================================================================================================

echo "======================"
echo "Post-Activation Script - Environment ${CONDA_DEFAULT_ENV}"
echo "======================"

# --- Imports --------------------------------------------------------------------------------------

# Root directory of the workspace (using the symlink in 'activate.d')

ACTIVATE_D="${CONDA_PREFIX}/etc/conda/activate.d"
export ROOT="$(readlink -f "$ACTIVATE_D/root")"
echo "Root: ${ROOT}"

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

ENV_FILE_META="${ROOT}/meta.env" # path to the meta.env file *relative to root*
if [ -f "${ENV_FILE_META}" ]; then
    set -a
    source "${ENV_FILE_META}"
    set +a
    echo "[SUCCESS] Set environment variables from '${ENV_FILE_META}'"
else
    echo "[ERROR] File not found at '${ENV_FILE_META}'."
    exit 1
fi

# Import Utils

if [ -f "${ENV_VAR_MNG}" ]; then
    source "${ENV_VAR_MNG}"
else
    echo "[ERROR] Utils not found at ${ENV_VAR_MNG}"
    exit 1
fi

if [ -f "${NAVIGATE_UTILS}" ]; then
    source "${NAVIGATE_UTILS}"
else
    echo "[ERROR] Utils not found at ${NAVIGATE_UTILS}"
    exit 1
fi

# --- Main Process ---------------------------------------------------------------------------------

# Navigate to the root directory
navigate_to_dir "${ROOT}"

# Add paths to the `PYTHONPATH` variable
add_from_file "${PTH_PYTHON}" "PYTHONPATH"
show_paths "PYTHONPATH"

# Add paths to the `PATH` variable
add_from_file "${PTH_BIN}" "PATH"
show_paths "PATH"

echo " === END ==="
