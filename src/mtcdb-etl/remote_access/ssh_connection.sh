#!/usr/bin/env bash

# ==================================================================================================
# Script Name: ssh_connection.sh
# Description: Orchestrates secure SSH connections to a remote server.
#
# Functions
# ---------
# connect           : Establishes an SSH connection with key authentication.
# connect_and_check : Verifies the key is sent before connecting.
#
# Requirements
# ------------
# - ssh_utils.sh must be sourced before calling connect or connect_and_check.
# - A valid .env file must be provided or located in the default config path.
#
# Usage
# -----
#   ./main.sh connect <path_to_credentials.env>
#
# ==================================================================================================

# Import utility functions
source "$(dirname "$0")/ssh_utils.sh"

# Function   : connect
# Description: Establishes an SSH connection using key-based authentication.
#
# Arguments
# ---------
# $1 : str
#   Path to the .env file containing the credentials for the remote server.
#
# Returns
# -------
#   0 if the connection is successful.
#   1 if the connection fails.
#
# Example
# -------
# .. code-block:: bash
#
#    connect ./config/credentials.env
#
connect() {
    local credentials="${1:-./config/credentials.env}"

    echo "[INFO] Loading credentials from $credentials"
    if ! load_credentials "$credentials"; then
        case $? in
            1) echo "[ERROR] Credentials file not found: $credentials" >&2 ;;
            2) echo "[ERROR] SSH_USER and SSH_HOST must be defined in the credentials file." >&2 ;;
            3) echo "[WARN] SSH_PATH_KEY is not defined. You must specify it for key-based login." >&2 ;;
        esac
        return 1
    fi

    echo "[INFO] Checking if SSH key is already configured..."
    if ! check_key_sent; then
        echo "[INFO] SSH key not found. Sending key now..."
        if send_keys; then
            echo "[INFO] SSH key successfully sent."
        else
            echo "[ERROR] Failed to send SSH key to $SSH_USER@$SSH_HOST." >&2
            return 1
        fi
    else
        echo "[INFO] SSH key already configured."
    fi

    echo "[INFO] Initiating SSH connection to $SSH_HOST as $SSH_USER..."
    if connect_with_key; then
        echo "[INFO] Successfully connected to $SSH_HOST."
        return 0
    else
        echo "[ERROR] Connection to $SSH_HOST failed." >&2
        return 1
    fi
}
