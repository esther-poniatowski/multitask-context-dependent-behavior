.PHONY: all help test docs clean export-env check-dev unregister-dev register-dev

help:
	@echo "Makefile commands:"
	@echo "   make test         	- Run tests and generate XML report for Sphinx"
	@echo "   make docs         	- Build Sphinx documentation in HTML"
	@echo "   make open-docs        - Open the HTML documentation in the default web browser"
	@echo "   make clean        	- Remove Python file artifacts"
	@echo "   make export-env   	- Export current conda environment to YAML"
	@echo "   make check-dev    	- Check packages registered in the environment in development mode"
	@echo "   make unregister-dev	- Unregister all packages currently in development mode (for resetting)"
	@echo "   make register-dev		- Register new packages in development mode : src and tests (for Sphinx extracting docstrings)"


test:
	pytest


docs:
	sphinx-build -b html docs/source/ docs/build/html

open-docs:
	@open docs/build/html/index.html

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -exec rm -f {} +


export-env:
	conda env export > environment.yml


check-dev:
	@$(eval SITE_PACKAGES=$(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"))
	@echo "Site-packages directory: $(SITE_PACKAGES)"
	@echo "Registered packages in development mode:"
	@cat $(SITE_PACKAGES)/conda.pth


unregister-dev:
	@cat $(shell python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/conda.pth | while read line; do \
		conda develop -u $$line; \
	done


register-dev:
	conda develop src
	conda develop tests