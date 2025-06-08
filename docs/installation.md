
# Installation Guide

## Downloading the Project

Download the project from the GitHub repository:

```sh
git clone https://github.com/esther-poniatowski/multitask-context-dependent-behavior.git
```

## Installation Approaches

The project can be installed using several methods, depending on the user's needs.

> [!NOTE]
> Is it recommended to create a dedicated virtual environment to install the project and its
> dependencies. In this guide, it is assumed that the project will be installed in a virtual
> environment (either `conda` or `venv`) named `mtcdb`.

### Using Pip

`pip` directly install the project from using the `pyproject.toml` file.

For runtime usage, install the project with its runtime dependencies only:

```sh
pip install .
```

For development, install the project in editable mode with all its optional dependencies:

```sh
pip install -e ".[dev]"
```

### Using Conda

`conda` cannot install the project directly since it is not distributed as a conda package. However,
it can be installed in two steps:


1. Create the environment containing the project's dependencies:

    For runtime usage:

    ```sh
    conda env create -name mtcdb -file environment.yml
    ```

    For development, select the `environment-dev.yml` file instead.

2. Register the package in the environment: This step will create a `.pth` file in the
    `site-packages` directory of the environment, allowing the Python interpreter to access the
    package for importing.

    For runtime usage:

    ```sh
    conda activate mtcdb
    echo "$(pwd)/src" > $(python -c "import site; print(site.getsitepackages()[0])")/mtcdb.pth
    ```

    For development, also register the test directory in the environment:

    ```sh
    echo "$(pwd)/tests" >> $(python -c "import site; print(site.getsitepackages()[0])")/mtcdb.pth
    ```

### Using Unidep

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
    unidep install . --name mtcdb --file environment.yml
    ```

This command performs the following actions in sequence:
    1. Installs all Conda installable dependencies.
    2. Installs remaining pip-only dependencies.
    3. Installs the local package in editable mode (using `pip install -e .`).

## Synchronizing Servers

ETL operations involve communication between two servers:

- **Local server**: The user's local machine, which orchestrates the ETL operations and initiates
  file transfers.
- **Remote server**: The central data hub, which is a secure storage location hosted by the
  laboratory (see [servers](docs/etl/servers.md)).

To synchronize those operations, the `data-etl` code has to be downloaded on both servers.

1. Install the project on the local server as described in sections `#downloading-the-project` and
   `#installation-approaches`.

2. Configure SSH keys for secure communication between the local and remote servers (see [SSH
   configuration guide](docs/etl/ssh_config.md)). Passwords can be provided by the project's owner
   upon request.

3. Connect to the secure data server: [connection-guide](docs/etl/servers.rst)

4. Clone the repository branch on this server as described in section `#downloading-the-project`.

5. Setup the working environment on the remote server:

   - Install the python dependencies as described in section `#installation-approaches`.

   - Install the MATLAB dependencies:

     ```sh

     ```

     <!-- TODO: Specify the command when available -->
