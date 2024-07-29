# =================================================================================================
# Makefile to deploy the relevant part of the workspace on the hub.
#
# Targets
# -------
# send-keys
# 	Send SSH keys to the hub.
# connect
# 	Connect to the data hub.
#
# Variables
# ---------
# ROOT
# 	Root directory of the workspace. Set upon activating the conda environment of the workspace.
# CREDENTIALS_DIR
# 	Directory containing the credentials.
# CREDENTIALS_ENV
# 	Environment file containing the credentials. For the hub: `lspaud1.env`.
#
# Warning
# -------
# In a Makefile recipe, each line is run in a *separate* subshell.
#
# Notes
# -----
# The := operator is used to assign the value of a variable at the time of the declaration.
#
# See Also
# --------
# send_keys.sh
# connect_with_keys.sh
# =================================================================================================

CONFIG_DIR := $(ROOT)/config
CREDENTIALS_DIR := $(CONFIG_DIR)/credentials
CREDENTIALS_ENV := $(CREDENTIALS_DIR)/lspaud1.env

.PHONY: send-keys
send-keys:
	$(CONFIG_DIR)/deploy/send_keys.sh $(CREDENTIALS_ENV)

.PHONY: connect
connect:
	$(CONFIG_DIR)/deploy/connect_with_key.sh $(CREDENTIALS_ENV)

.PHONY: check-inclusion
check-inclusion:
	@echo "ROOT: $(ROOT)"
	@echo "CONFIG_DIR: $(CONFIG_DIR)"
	@echo "CREDENTIALS_DIR: $(CREDENTIALS_DIR)"
	@echo "CREDENTIALS_ENV: $(CREDENTIALS_ENV)"


mount-data:
	# TODO: Run the script mount_data.sh
