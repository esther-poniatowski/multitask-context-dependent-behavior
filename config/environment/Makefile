# =================================================================================================
# Makefile to manage the virtual environment of the project.
#
# Targets
# -------
# clean
# 	Remove temporary files.
# export-env
# 	Export the environment to a file.
# check-dev
# 	Check the packages registered in development mode.
# unregister-dev
# 	Unregister the packages in development mode.
# register-dev
# 	Register the packages in development mode.
# =================================================================================================


.PHONY: clean
clean:
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -exec rm -f {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +

.PHONY: export-env
export-env:
	conda env export > config/environment.yml

.PHONY: check-dev
check-dev:
	@$(eval SITE_PACKAGES=$(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"))
	@echo "Site-packages directory: $(SITE_PACKAGES)"
	@echo "Registered packages in development mode:"
	@cat $(SITE_PACKAGES)/conda.pth

.PHONY: unregister-dev
unregister-dev:
	@cat $(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/conda.pth | while read line; do \
		conda develop -u $$line; \
	done

.PHONY: register-dev
register-dev:
	conda develop src
	conda develop tests
