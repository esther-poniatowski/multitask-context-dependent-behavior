#!/usr/bin/env bash

# ==================================================================================================
# Script Name: ssh_utils.sh
# Description: Manage secure connections to a remote server.
#
# Functions
# ---------
# load_credentials      : Load the credentials from a .env file.
# send_keys             : Send an SSH key to a remote server.
# check_key_sent        : Ensure that the SSH key has already been sent to the remote server.
# connect_with_key      : Connect to the remote server using the SSH key pair.
#
# Requirements
# ------------
# SSH client tools installed (ssh, ssh-copy-id, ssh-agent).
# Credentials file in .env format defining SSH_USER, SSH_HOST, and optionally SSH_PASS.
#
# Usage
# -----
#   sudo ./connect.sh <path_to_credentials.env>
#
# ==================================================================================================


# Function   : load_credentials
# Description: Load the credentials from a .env file and validate them.
#
# Arguments
# ---------
# $1 : str
#   Path to the .env file containing the credentials for the remote server.
#
# Returns
# -------
#   0 if the credentials are loaded successfully.
#   1 if the credentials file is provided but does not exist.
#   2 if required variables are missing.
#   3 if optional variables are missing.
#
# Example
# -------
# .. code-block:: bash
#
#    load_credentials "path/to/credentials.env"
#
load_credentials() {
    local credentials=$1
    # If credentials file provided, load it
    if [ -n "$credentials" ] && [ -f "$credentials" ]; then
        set -a
        source "$credentials"
        set +a
    else
        return 1
    fi
    # Validate required variables exported in the environment
    if [ -z "${SSH_USER:-}" ] || [ -z "${SSH_HOST:-}" ]; then
        return 2
    fi
    # Check optional password variable
    if [ -z "${SSH_PASS:-}" ]; then
        return 3
    fi
    return 0
}

# Function   : send_keys
# Description: Send an SSH key to a remote server.
#
# Returns
# -------
#   0 if the key is sent successfully.
#   1 if the key sending fails.
#
# Example
# -------
# .. code-block:: bash
#
#    send_keys
#
send_keys() {
    ssh-copy-id -q "$SSH_USER@$SSH_HOST"
    return $?
}

# Function   : check_key_sent
# Description: Ensure that an SSH key is configured on a remote server.
#
# Returns
# -------
#   0 if the key is already sent.
#   1 if the key is not sent.
#
# Example
# -------
# .. code-block:: bash
#
#    check_key_sent
#
check_key_sent() {
    ssh -o PasswordAuthentication=no \
        -o BatchMode=yes \
        -o ConnectTimeout=1 \
        -q "$SSH_USER@$SSH_HOST" true
    return $?
}

# Function   : connect_with_key
# Description: Connect to a remote server using an SSH key pair.
#
# Returns
# -------
#   0 if the connection is successful.
#   1 if the connection fails.
#   2 if the SSH_KEY_PATH variable is not set or the file does not exist.
#
# Example
# -------
# .. code-block:: bash
#
#    connect_with_key
#
# TODO: Is it the best way to pass a key to the agent?
connect_with_key() {
    if [ -z "${SSH_KEY_PATH:-}" ] || [ ! -f "$SSH_KEY_PATH" ]; then
        return 2
    fi
    # Check if ssh-agent is running and start it if not
    if ! pgrep -u "$USER" ssh-agent > /dev/null; then
        eval "$(ssh-agent -s)"
    fi
    # Add the SSH key to the agent if not already added
    ssh-add -l | grep -q "$SSH_KEY_PATH" || ssh-add "$SSH_KEY_PATH"
    # Attempt to connect to the remote server
    ssh -X "$SSH_USER@$SSH_HOST"
    return $?
}


# Function : cleanup_agent
# Description: Clean up the SSH agent after use.
#
# Returns
# -------
#   0 if the agent is cleaned up successfully.
#   1 if there is no agent to clean up.
#
# Example
# -------
# .. code-block:: bash
#
# cleanup_agent
#
cleanup_agent() {
    if pgrep -u "$USER" ssh-agent > /dev/null; then
        ssh-agent -k
        return 0
    else
        return 1
    fi
}