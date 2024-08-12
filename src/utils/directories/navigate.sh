#!/usr/bin/env bash

# ==================================================================================================
# Script Name:   navigate.sh
# Description:   Utilities to navigate across the directory structure in the shell session.
#
# Functions
# ---------
# navigate_to_dir
#
# ==================================================================================================


# Function:    navigate_to_dir
# Description: Set the current working directory for the shell session.
#
# Arguments
# ---------
# dir: str
#     Path to the target directory which will become be the current working directory.
#
# Raises
# ------
# [ERROR] If the directory does not exist or navigation fails.
#
# Usage
# -----
# .. code-block:: bash
#
#    navigate_to_dir "/path/to/target/directory"
#
navigate_to_dir() {
    local dir="$1"
    cd "$dir" || {
        echo "[ERROR] Failed to navigate to $dir"
        exit 1
    }
    echo "[SUCCESS] Navigated to $dir"
}
