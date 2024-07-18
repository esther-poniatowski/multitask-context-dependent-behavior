# Import passwords and IP addresses
include .env

.PHONY: send-keys connect_biggerguy mount_volume connect_lsp send_codes_to_lsp send_codes_to_biggerguy generate_site_lists copy_site_lists

# Directory containing the transfer codes
ROOT_PATH := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DIR_NAME := $(shell basename $(ROOT_PATH))

help:
	@echo "Makefile commands:"
	@echo "

send-keys:
	ssh-copy-id $(USER_LSPAUD1)@$(IP_LSPAUD1); echo $(PASS_LSPAUD1)
	ssh-copy-id $(USER_MARRSFREESTYLE)@$(IP_MARRSFREESTYLE); echo $(PASS_MARRSFREESTYLE)

connect-lspaud1:
	eval "$(ssh-agent -s)"
	ssh-add $(PATH_KEY)
	ssh -X $(USER_LSPAUD1)@$(IP_LSPAUD1)

connect-marrsfreestyle:
	ssh $(USER_MARRSFREESTYLE)@$(IP_MARRSFREESTYLE)

mount-data4:
	ssh $(USER_LSPAUD1)@$(IP_LSPAUD1) 'cd; sudo mount -t cifs -o username=$(USER_LSPAUD1),domain=LSPneuro_NAS1,rw,noperm,gid=$(USER_LSPAUD1),iocharset=utf8 //$(IP_DATA4)/data4 ~/data4; echo $(PASS_DATA4)'
