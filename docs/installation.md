
# Installation

The project can be installed using several methods, depending on the user's needs.

## Downloading the Project

Download the project from the GitHub repository:

```sh
git clone https://github.com/esther-poniatowski/multitask-context-dependent-behavior.git
```

## Installing via Pip

`pip` directly install the project from using the `pyproject.toml` file.

For runtime usage, install the project with its runtime dependencies only:

```sh
pip install .
```

For development, install the project in editable mode with all its optional dependencies:

```sh
pip install -e ".[dev]"
```

## Installing via Conda

`conda` cannot install the project directly since it is not distributed as a conda package. However,
it can be installed in two steps:


1. Create the environment containing the project's dependencies:

    For runtime usage:

    ```sh
    conda env create -name mtcdb-etl -file environment.yml
    ```

    For development, select the `environment-dev.yml` file instead.

2. Register the package in the environment: This step will create a `.pth` file in the
    `site-packages` directory of the environment, allowing the Python interpreter to access the
    package for importing.

    For runtime usage:

    ```sh
    conda activate mtcdb-etl
    echo "$(pwd)/src" > $(python -c "import site; print(site.getsitepackages()[0])")/mtcdb-etl.pth
    ```

    For development, also register the test directory in the environment:

    ```sh
    echo "$(pwd)/tests" >> $(python -c "import site; print(site.getsitepackages()[0])")/mtcdb-etl.pth
    ```

## Installing via Unidep

The project has been configured to use [`unidep`](https://unidep.readthedocs.io/en/latest/) to
synchronize dependencies between `pip` and `conda`. While this tool is not necessary for installing
the project, it facilitates the creation of a conda environment.

1. Install `unidep`:

    ```sh
    pip install unidep
    ```

    or

    ```sh
    conda install -c conda-forge unidep
    ````

2. Create the conda environment with all the dependencies:

    ```sh
    unidep install .
    ```

This command performs the following actions in sequence:
    1. Installs all Conda installable dependencies.
    2. Installs remaining pip-only dependencies.
    3. Installs the local package in editable mode (using `pip install -e .`).
