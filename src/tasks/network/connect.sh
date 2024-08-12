#!/usr/bin/env bash

# ==================================================================================================
# Script Name:   connect.sh
# Description:   Manage connections to a remote server.
#
# Functions
# ---------
# send_keys
# connect_with_key
#
# ==================================================================================================


# Function:    send_keys
# Description: Send a SSH key to a remote server.
#
# Arguments
# ---------
# credentials : str
#   Path to the .env file containing the credentials:
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
    # Load the credentials from the .env file
    source "${credentials}" || {
        echo "[ERROR] Failed to load the credentials."
        exit 1
    }
    # Display connection information
    echo "User: $USER"
    echo "Host: $HOST"
    # Send the SSH keys to the remote server
    ssh-copy-id $USER@$HOST && echo $PASS || {
        echo "[ERROR] Failed to send SSH keys to the remote server."
        exit 1
    }
    echo "[SUCCESS] SSH keys sent to the remote server."
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
# [ERROR] If the connection fails.
#
# Example
# -------
# .. code-block:: bash
#
#    connect_with_key "path/to/credentials.env"
#
connect_with_key() {
    local credentials=$1
    # Load the credentials from the .env file
    source "${credentials}" || {
        echo "[ERROR] Failed to load the credentials."
        exit 1
    }
    # Display connection information
    echo "User: $USER"
    echo "Host: $HOST"
    # Start the SSH agent
    eval "$(ssh-agent -s)"
    # Add the SSH key to the agent
    ssh-add $PATH_KEY
    # Connect to the remote server
    ssh -X $USER@$HOST || {
        echo "[ERROR] Failed to connect to the remote server."
        exit 1
    }
    echo "[SUCCESS] Connected to the remote server."
}
