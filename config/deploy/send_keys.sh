#!/usr/bin/env bash

# ============================================================================
# Script Name:   send_keys.sh
# Description:   Send the SSH key to the remote server.
#
# Arguments
# ---------
# CREDENTIALS:
#   Path to the .env file containing the credentials.
# =============================================================================

# Load the credentials from the .env file passed as argument
CREDENTIALS=$1
source $CREDENTIALS

ssh-copy-id $USER@$HOST
echo $PASS
