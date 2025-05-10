# Multi-Task Context-Dependent Decision Making - ETL

## Overview

This `data-etl` branch is part of the Multi-Task Context-Dependent Decision Making project,
dedicated to the extraction, transformation, and loading (ETL) stages, prior to analyses.

Those actions operate on secure data servers owned by the laboratory's department.

> [!IMPORTANT] This branch is maintained separately from the `main` branch of the project
> repository, to isolate the ETL code so that it can be downloaded and run independently of the main
> project. It is not meant to be merged into the main branch.

## Installation

### Runtime Environment

ETL operations involve communication between two servers:

- **Local server**: The user's local machine, which orchestrates the ETL operations and initiates
  file transfers.
- **Remote server**: The central data hub, which is a secure storage location hosted by the
  laboratory (see [servers](docs/servers.rst)).

To synchronize those operations, the `data-etl` code has to be downloaded on both servers.

1. Clone the `data-etl` branch of project's repository:

   ```sh
   git clone --recurse-submodules --single-branch --branch data-etl https://github.com/esther-poniatowski/multitask-context-dependent-behavior.git
   ```

2. Create a virtual environment dedicated to the project, for instance named `mtcdb-etl`:

   ```sh
   conda env create --name mtcdb-etl --file conda-lock.yml
   ```

3. Configure SSH keys for secure communication between the local and remote servers (see [SSH
   configuration guide](docs/ssh_config.rst)). Passwords can be provided by the project's owner upon
   request.

4. Connect to the secure data server: [connection-guide](docs/servers.rst)

5. Clone the `data-etl` branch on this server as in step 1.

6. Setup the working environment on the remote server:
   - Install the python dependencies:

     ```sh
     conda env create --name mtcdb-etl --file conda-lock.yml
     ```

   - Install the MATLAB dependencies:

     ```sh
     TODO: specify the command when available
     ```

> [!NOTE] Cloning the `data-etl` branch will automatically include the `janux` package as a Git
> submodule in the `include/` folder (as specified in the `.gitmodules` file). This package is
> responsible for setting up SSH connections between the local and remote servers.

### Development Environment

> [!WARNING] This step requires the `unidep` tool to be installed.

To develop the ETL code, the `data-etl` branch:

1. Perform step 1 of the runtime environment installation.

2. Generate development-specific lock files from the `pyproject.toml` files of the super and
   submodules, using the `conda-lock` package:

   ```sh
   unidep merge
   unidep condalock
   ```


## Workflow

The early stages of the project aim to:

1. Collect data from diverse sources and collaborators (Maryland servers, personal transmission
   channels) and organize it on a central data server:
   [data-collection-guide](docs/data_collection.rst).

2. Organize files in a logical directory tree appropriate for the subsequent analyses:
   [file-system](docs/file_system.rst)

3. Unpack specific content from raw files (MATLAB: `.m`, `.mat`, or compressed archives: `.zip`,
   `.tar.gz`) and export it to universal formats (`.csv`), that are compatible with Python:
   [data-extraction-guide](docs/data_extraction.rst)

4. Transfer selected data to a server were downstream analysis will be performed:
   [data-transfer-guide](docs/data_transfer.rst)

## Directory Structure

### Top-Level

```plaintext
├── config/          
├── data/
├── docs/
├── src/
├── environment.yml
├── LICENSE
├── main.sh
├── mtcdb.code-workspace
├── README.md
└── ROADMAP.md
```

## License

This project is licensed under the [GPL License](LICENSE).
