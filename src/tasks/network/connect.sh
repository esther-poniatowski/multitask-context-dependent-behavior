#!/usr/bin/env bash

# ==================================================================================================
# Script Name:   connect.sh
# Description:   Manage connections to a remote server.
#
# Functions
# ---------
# load_credentials
# send_keys
# check_key_sent
# connect_with_key
#
# ==================================================================================================

# Function:    load_credentials
# Description: Load the credentials from a .env file.
#
# Arguments
# ---------
# credentials : str
#   Path to the .env file containing the credentials for the remote server.
#
# Raises
# ------
# [ERROR] If the credentials cannot be loaded.
#
# Example
# -------
# .. code-block:: bash
#
#    load_credentials "path/to/credentials.env"
#
load_credentials() {
    local credentials=$1
    # Load the credentials from the .env file
    source "${credentials}" || {
        echo "[ERROR] Failed to load the credentials."
        exit 1
    }
    # Display connection information
    echo "User: $USER"
    echo "Host: $HOST"
}


# Function:    send_keys
# Description: Send a SSH key to a remote server.
#
# Arguments
# ---------
# credentials : str
#   Path to the .env file containing the credentials for the remote server:
#
#   - USER : Username to connect to the remote server.
#   - HOST : Hostname of the remote server.
#   - PASS : Password to connect to the remote server.
#
# Raises
# ------
# [ERROR] If the keys cannot be sent.
#
# Example
# -------
# .. code-block:: bash
#
#    send_keys "path/to/credentials.env"
#
send_keys() {
    local credentials=$1
    load_credentials "$credentials"
    ssh-copy-id "$USER@$HOST" && echo "$PASS" || {
        echo "[ERROR] Failed to send SSH keys to the remote server."
        exit 1
    }
    echo "[SUCCESS] SSH keys sent to the remote server."
}


# Function:    check_key_sent
# Description: Ensure that the SSH key has already been sent to a remote server.
#
# Arguments
# ---------
# credentials : str
#   Path to the .env file containing the credentials for the remote server.
#
# Example
# -------
# .. code-block:: bash
#
#    check_key_sent "path/to/credentials.env"
#
# Notes
# -----
# To check if the SSH key exists on the remote server, the function tries to connect to the server
# using the key in a non-unteractive mode.
# If the SSH connection is successful (indicating the key is properly set up), then the connection
# is immediately closed and a message is printed to the console.
# If the SSH connection fails (indicating the key is not properly set up), then an error message is
# printed and status code of 1 is returned, indicating an error.
#
# Implementation
# --------------
# - `-o`: Set an option for the SSH client.
# - `PasswordAuthentication=no`: Disable password authentication.
# - `BatchMode=yes`: Disable any interactive prompts (requirements for passwords or passphrases).
# - `ConnectTimeout=1`: Set the timeout for connecting to the remote server to 1 second.
# - `-q`: Quiet mode, suppress most warnings and diagnostic messages.
# - `true`: Immediately close the connection after a successful login, with a successful status.
# - `return`: Return a status rather than exiting the entire script.
#
check_key_sent() {
    local credentials=$1
    load_credentials "$credentials"
    if ssh -o PasswordAuthentication=no -o BatchMode=yes -o ConnectTimeout=1 -q "$USER@$HOST" true ; then
        echo "[INFO] SSH key is properly set up on the remote server."
        return 0
    else
        echo "[ERROR] SSH key not found or not properly configured on the remote server."
        return 1
    fi
}


# Function:    connect_with_key
# Description: Connect to a remote server using a SSH key pair.
#
# Arguments
# ---------
# credentials : str
#   Path to the .env file containing the credentials:
#
#   - USER : Username to connect to the remote server.
#   - HOST : Hostname of the remote server.
#   - PATH_KEY : Path to the SSH private key.
#
# Raises
# ------
# [ERROR] If connection fails.
#
# Example
# -------
# .. code-block:: bash
#
#    connect_with_key "path/to/credentials.env"
#
connect_with_key() {
    local credentials=$1
    load_credentials "$credentials"
    # Start the SSH agent
    eval "$(ssh-agent -s)"
    # Add the SSH key to the agent
    ssh-add "$PATH_KEY"
    # Connect to the remote server
    ssh -X "$USER@$HOST" || {
        echo "[ERROR] Failed to connect to the remote server."
        exit 1
    }
    echo "[SUCCESS] Connected to the remote server."
}
