#!/usr/bin/env bash

# ==================================================================================================
# Script Name:   env_var_mng.sh
# Description:   Manage environment variables in the shell session.
#
# Functions
# ---------
# export_env_vars
# show_paths
# add_path
# add_from_file
#
# Warning
# -------
# Do NOT use `path` as a variable name.
# In the `zsh`` shell, this variable is reserved for an array representation of the system `PATH`
# variable and is automatically synchronized with the `PATH` variable.
# ==================================================================================================


# Function:    export_env_vars
# Description: Set environment variables from a ``.env`` file in the current shell session.
#
# Arguments
# ---------
# env_file: str
#     Path to the ``.env`` file that contains the environment variables to be sourced.
#
# Raises
# ------
# [ERROR] If the specified ``.env`` file is not found.
#
# Example
# -------
# .. code-block:: bash
#
#    export_env_vars "/path/to/.env"
#
# Notes
# -----
# Variables are exported in the shell session by sourcing the `.env` file. This approach handles
# complex variable assignments. To make the variable available in any child processes, the shell
# behavior has to be modified using the `set -a` and `set +a` commands. The `set -a` command enables
# the export of all variables, which is not the default behavior (whereas it is automatic via the
# `export` command).
#
export_env_vars() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        set -a
        source "$env_file"
        set +a
        echo "[SUCCESS] Set environment variables from $env_file."
    else
        echo "[ERROR] Failed to find .env file at $env_file."
        exit 1
    fi
}


# Function:    show_paths
# Description: Display paths in an environment variable.
#
# Arguments
# ---------
# env_var: str
#     Name of the environment variable which stores the paths.
#
# Raises
# ------
# Warning: If the environment variable is empty or not set.
#
# Example
# -------
# To display the paths in the `PYTHONPATH` environment variable:
#
# .. code-block:: bash
#
#    show_paths "PYTHONPATH"
#
# Notes
# -----
# Contents of the local variables:
#
# - `env_var`  : *Name* of the environment variable (string).
# - `env_value`: *Value* of the environment variable.
#
# Commands:
#
# - `\$$env_var` : Construct a string that stores the command to retrieve the value of the
#   environment variable whose name is stored in `env_var`.
# - `eval` : Evaluate the string as a shell command to assign the value of the environment variable.
# - `sed 's|:|\n|g'`  : Replace colon delimiters by a newline character globally (`g`).
# - `sed 's|^|   - |'`: Apply a prefix to the start of each line (^).
#
show_paths() {
    local env_var="$1"
    local env_value
    eval env_value=\$$env_var
    echo "Paths in $env_var:"
    if [[ -n "$env_value" ]]; then
        echo "$env_value" | sed 's|:|\n|g' | sed 's|^|   - |'
    else
        echo "[WARNING] Empty or not set."
    fi
}


# Function:    add_path
# Description: Prepend a new path to an environment variable. Add execution rights.
#
# Arguments
# ---------
# path_to_add: str
#     Path to be added to the environment variable, if not already present.
# env_var: str
#     Name of the environment variable to which the path will be added.
#
# Example
# -------
# To add a directory to the PYTHONPATH environment variable:
#
# .. code-block:: bash
#
#    add_path "/path/to/directory" "PYTHONPATH"
#
add_path() {
    local path_to_add="$1"
    local env_var="$2"
    local env_value
    eval env_value=\$$env_var
    if [[ -z "$env_value" ]]; then  # if empty, directly set to the new path
        export $env_var="$path_to_add"
    elif [[ ":$env_value:" != *":$path_to_add:"* ]]; then # if not already present, prepend new path
        export $env_var="$path_to_add:$env_value"
    fi
    # Set execution rights to the directory
    chmod -R +x "$path_to_add" || {
        echo "[WARNING] Failed to set execution rights to $path_to_add"
    }
    echo "[SUCCESS] $env_var includes $path_to_add"
}


# Function:    add_from_file
# Description: Prepend to an environment variable a set of paths stored in a .pth file.
#              Add execution rights.
#
# Arguments
# ---------
# file_path: str
#     Path to the .pth file (plaintext) containing paths to be added.
# env_var: str
#     Name of the environment variable to which paths will be added.
#
# Example
# -------
# To add paths from a file to the PYTHONPATH environment variable:
#
# .. code-block:: bash
#
#    add_from_file "/path/to/python.pth" "PYTHONPATH"
#
# See Also
# --------
# add_path: Function to add a new path to an environment variable.
#
add_from_file() {
    local file_path="$1"
    local env_var="$2"
    local line
    local path_string
    if [ -f "$file_path" ]; then  # check file existence
        echo "Processing file: $file_path"
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Trim leading/trailing whitespace
            path_string=$(echo "$line" | xargs)
            # Skip empty lines, whitespace-only lines, and comment lines
            if [[ -n "$path_string" && ! "$path_string" =~ ^# ]]; then
                # Expand variables
                expanded_path=$(eval echo "$path_string")
                # Add path to the environment variable
                add_path "$expanded_path" "$env_var"
                # Set execution rights to the directory
                chmod -R +x "$expanded_path" || {
                    echo "[WARNING] Failed to set execution rights to $expanded_path"
                }
            fi
        done < "$file_path"
    else
        echo "[WARNING] File not found at $file_path"
    fi
}
