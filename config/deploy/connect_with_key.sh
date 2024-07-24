#!/usr/bin/env bash

# ============================================================================
# Script Name:   connect_with_key.sh
# Description:   Connect to a remote server using a SSH key pair.
#
# Arguments
# ---------
# CREDENTIALS:
#   Path to the .env file containing the credentials.
#
# Notes
# -----
# Steps:
# 1. Load the credentials from the .env file.
# 2. Start the SSH agent.
# 3. Add the SSH key to the agent.
# 4. Connect to the remote server.
# =============================================================================

# Load the credentials from the .env file passed as argument
CREDENTIALS=$1
source $CREDENTIALS

eval "$(ssh-agent -s)"
ssh-add $(PATH_KEY)
ssh -X $(USER)@$(HOST)
