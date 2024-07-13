.PHONY: all help test docs clean export-env check-dev unregister-dev register-dev

help:
	@echo "Makefile commands:"
	@echo "   make format        	- Format Python code with Black"
	@echo "   make lint          	- Lint Python code with MyPy, Pylint and Pyright"
	@echo "   make test         	- Run tests and generate XML report for Sphinx"
	@echo "   make docs         	- Build Sphinx documentation in HTML"
	@echo "   make open-docs        - Open the HTML documentation in the default web browser"
	@echo "   make clean        	- Remove Python file artifacts"
	@echo "   make export-env   	- Export current conda environment to YAML"
	@echo "   make check-dev    	- Check packages registered in the environment in development mode"
	@echo "   make unregister-dev	- Unregister all packages currently in development mode (reset)"
	@echo "   make register-dev		- Register new packages in development mode : src and tests"
	@echo "   make setup-dev		- Set up the environment in developing mode"


format:
	black --config config/black.toml --diff src tests

lint:
	mypy --config-file config/mypy.ini src tests
	pylint --rcfile config/pylintrc src tests
	pyright --config-file config/pyrightconfig.json src tests

test:
	pytest

docs:
	sphinx-build -b html docs/source/ docs/build/html
	@open docs/build/html/index.html

open-docs:
	@open docs/build/html/index.html

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -exec rm -f {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +


export-env:
	conda env export > config/environment.yml


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

setup-dev:
