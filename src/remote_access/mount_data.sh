#!/usr/bin/env bash

# ==================================================================================================
# Script Name: `mount_data.sh`
# Description: Mount the CIFS `data4` from the NAS to the local directory.
#
# Notes
# -----
# A CIFS stands for Common Internet File System, which is a protocol that allows sharing files and printers
# over a network. It is commonly used in Windows environments but can also be used on Linux systems.
# A NAS (Network Attached Storage) is a dedicated file storage device that provides local area network (LAN) users
# with centralized, shared data storage through a standard Ethernet connection.
# The CIFS protocol allows users to access files on the NAS as if they were on their local machine.
# Syntax of the `mount` command:
#
# .. code-block:: bash
#
#    sudo mount -t <file_system_type> -o <options> <device_or_remote_share> <mount_point>
#
# Example for a CIFS share:
#
# .. code-block:: bash
#
#    sudo mount -t cifs\
#         -o username=<username>,\
#         domain=<domain>,\
#         rw,\
#         noperm,\
#         gid=<group_id>,\
#         iocharset=utf8 \
#         //<host>/<share_name> <local_dir>
#
# Options used:
# - `username`               : Username used to access the CIFS share.
# - `domain`                 : Name of the computer where the CIFS share is located.
# - `rw`                     : Mount with read and write permissions.
# - `noperm`                 : Disable permission checks.
# - `gid`                    : Group ID for the mounted filesystem.
# - `iocharset`              : Character encoding.
# - `//host/share_name`      : Network path to the CIFS share.
# - `local_dir`              : Local directory where the CIFS share will be mounted.
# Syntax `//`: Indicate that the path is a *network* location (not a *local* file path).
# ==================================================================================================

# Load Variables from credential file TODO: Make it an optional argument of the script
PATH_CONFIG="$ROOT/config/data4.env"
source $PATH_CONFIG

# Mount command
sudo mount -t cifs \
    -o username=$USER,\
       domain=$DOMAIN,\
       rw,\
       noperm,\
       gid=$USER,\
       iocharset=utf8 \
    //$HOST/data4 ~/data4

# Check execution status
if [ $? -eq 0 ]; then
    echo $PASS
    echo "Mount successful"
else
    echo "Mount failed"
fi
