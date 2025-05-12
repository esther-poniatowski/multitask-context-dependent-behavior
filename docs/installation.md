
# Installation

## Downloading the Project

Download the project from the GitHub repository:

```sh
git clone https://github.com/esther-poniatowski/multitask-context-dependent-behavior.git
```

## Installing via Pip

For runtime usage, install the project with its runtime dependencies only:

```sh
pip install .
```

For development, install the project in editable mode with all its optional dependencies:

```sh
pip install -e ".[dev]"
```

## Installing via Conda

For runtime usage:

1. Create the environment with the runtime dependencies only:

    ```sh
    conda env create -name mtcdb-etl -file environment.yml
    ```

2. Register the package in the environment:

    ```sh
    conda activate mtcdb-etl
    echo "$(pwd)/src" > $(python -c "import site; print(site.getsitepackages()[0])")/mtcdb-etl.pth
    ```

    This step will create a `.pth` file in the `site-packages` directory of the environment,
    allowing the Python interpreter to access the package for importing.

For development:

1. Create the environment with all the optional dependencies:

    ```sh
    conda env create -name mtcdb-etl -file environment-dev.yml
    ```

2. Register the package and the test directory in the environment:

    ```sh
    conda activate mtcdb-etl
    echo "$(pwd)/src" > $(python -c "import site; print(site.getsitepackages()[0])")/mtcdb-etl.pth
    echo "$(pwd)/tests" >> $(python -c "import site; print(site.getsitepackages()[0])")/mtcdb-etl.pth
    ```
