#!/usr/bin/env bash

# ==================================================================================================
# TODO: Adapt this script in Python
# Script Name: main.sh
# Description: Main entry script for managing remote access operations.
#
# Usage
# -----
#   ./main.sh <operation> [args...]
#
# Options
# -------
#  operation : Keyword specifying the operation to perform:
#     - connect : Establishes an SSH connection
#     - help    : Displays usage information
#
# Usage
# -----
#   ./main.sh connect <path_to_credentials.env>
#
# ==================================================================================================

# === FUNCTIONS ====================================================================================

# Function   : show_help
# Description: Displays the help message for the script.
show_help() {
    cat <<EOF
Usage: ./main.sh <operation> [args...]

Operations:
  connect   - Establishes an SSH connection
              Usage: ./main.sh connect <path_to_credentials.env>
  help      - Displays this help message
EOF
}

# Function   : dispatch
# Description: Dispatches the operation to the appropriate handler.
dispatch() {
    local operation=$1
    shift

    case "$operation" in
        connect)
            source "$DIR/src/remote_access/ssh_connection.sh"
            connect "$@"
            ;;
        help)
            show_help
            ;;
        *)
            echo "[ERROR] Unknown operation: $operation. Use './main.sh help' for usage." >&2
            exit 1
            ;;
    esac
}

# === EXECUTION ====================================================================================

set -euo pipefail

# Get the absolute path of the script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Validate input
if [ "$#" -lt 1 ]; then
    echo "[ERROR] Operation not specified. Use './main.sh help' for usage." >&2
    exit 1
fi

# Main execution
dispatch "$@"


# Validate input
if [ "$#" -lt 1 ]; then
    echo "[ERROR] Operation not specified. Use './main.sh help' for usage."
    exit 1
fi
