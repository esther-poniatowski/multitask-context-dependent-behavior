#!/usr/bin/env bash

# ============================================================================
# Script Name:   tree.sh
# Description:   Mimic the 'tree' command to display directory structure.
# Author:        Esther Poniatowski
# Created:       2024-07-12
# Last Modified: 2024-07-12
# Version:       1.0
# License:       MIT
# ============================================================================

# Set strict mode
set -euo pipefail

# Script metadata
readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_VERSION="1.0.0"

# === Function Definitions ===================================================
# Function: usage
# Description: Display usage information.
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] directory

Mimic the 'tree' command to display directory structure.

Options:
  -h, --help     Show this help message and exit
  -I pattern     Ignore paths that match the pattern
  -L level       Maximum depth of the directory tree to display
  -N number      Maximum number of items to display per directory # TODO

Arguments:
  directory      Root directory to display

Examples:
  Limit to 3 levels :
    $SCRIPT_NAME -L 3 /path/to/directory
  Ignore all the files containing the pattern ".ignore" :
    $SCRIPT_NAME -I "*ignore*" /path/to/directory
  Ignore all the files in the directories "tests" and "docs" :
    $SCRIPT_NAME -I "tests|docs" /path/to/directory
EOF
    exit 1
}

# Function:    print_tree
# Description: Print the directory tree structure.
#                   It prints the directory structure with the appropriate
#                   prefix  and pointer (│, └──) characters.


# Parameters:
#   $1 - Directory to start from
#   $2 - Prefix for the tree branches (built from │, ├──, └──)
#   $3 - Current depth level
#   $4 - Pattern in items to ignore
#   $5 - Max depth level
# Returns:
#   None
# Notes:
# Use recursion to traverse the directory tree.
# Stop recursion when the max depth level is reached.
# This is implemented at lines
print_tree() {
    local dir=$1
    local prefix=$2
    local depth=$3
    local ignore_pattern=$4
    local level=$5

    # Iterate over the items in the directory
    local items=($(ls -A "$dir"))
    local count=${#items[@]} # max index
    # `@` : select all the elements of the array
    # `#` : get the length of the array

    for ((i = 0; i < count; i++)); do

        local item=${items[$i]}

        # Skip items that match the ignore pattern
        if [[ $item == $ignore_pattern ]]; then
            continue
        fi

        # Set the pointer character
        local pointer="├── " # default for non-last items
        if [ $i -eq $((count - 1)) ]; then # change for last item
            pointer="└── "
        fi

        # Print the item with the appropriate prefix and pointer
        echo "${prefix}${pointer}${item}"

        # Recursively call the function for subdirectories
        local path="$dir/$item"
        if [ -d "$path" ]; then
            # Set the extension character
            local extension="│   " # default for non-last items
            if [ $i -eq $((count - 1)) ]; then # change for last item
                extension="    "
            fi
            print_tree "$path" "$prefix$extension" $((depth + 1)) "$ignore_pattern" $((level - 1))
        fi
    done
}


# Function:    main
# Description: Main function of the script.
main() {
    local directory=$1

    echo "Directory Tree: $directory"

    # Validate directory
    if [ ! -d "$directory" ]; then
        echo "Error: '$directory' is not a directory." >&2
        exit 1
    fi

    # Print the tree structure
    echo "==="
    print_tree "$directory" "" 1 "$ignore_pattern" "$max_depth"
    echo "==="
}

# === Main Script ============================================================
# Default values for variables
ignore_pattern=""
max_depth=-1
directory="."

# Parse command-line options
while getopts ":I:L:h" opt; do
    case $opt in
        I)
            ignore_pattern=$OPTARG
            ;;
        L)
            max_depth=$OPTARG
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

# Check if directory is provided
if [ $# -ne 1 ]; then
    echo "Error: Missing required directory argument." >&2
    usage
fi

# Call the main function with the directory argument
main "$1"

# Suggestion for code change
./server-transfer/tree.sh .

# Suggestion for code change
./server-transfer/tree.sh .
