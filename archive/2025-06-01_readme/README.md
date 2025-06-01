# Multi-Task Context-Dependent Decision Making

**Table of Contents**:

- [Workflow](#workflow)
  - [Extraction, Transformation, Loading (ETL)](#extraction-transformation-loading-etl)
  - [Data Analysis](#data-analysis)
- [Packages](#packages)
- [Directory Structure](#directory-structure)
- [Data Storage](#data-storage)

**Full Documentation**: [multitask-context-dependent-behavior](https://esther-poniatowski.github.io/multitask-context-dependent-behavior/)

**Esther Poniatowski** | @esther-poniatowski | esther.poniatowski@ens.psl.eu

## Workflow

### Extraction, Transformation, Loading (ETL)

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

### Data Analysis

---

## Project Structure

### Packages

This `data-etl` package is part of the Multi-Task Context-Dependent Decision Making project,
dedicated to the extraction, transformation, and loading (ETL) stages, prior to analyses.

Those actions operate on secure data servers owned by the laboratory's department.

> [!IMPORTANT]
> This package is maintained separately from the main analysis project to isolate the ETL code.
> This package can be downloaded and run independently of the analysis.

### Directory Structure

The project structure separates functionality/logic ("how") from execution/task runners ("what" and
"when"). For chore tasks, if these are complex or involve multiple steps, they can be encapsulated
in separate modules/classes within the `tasks/` directory. This way, the `ops/` scripts can import
and execute these tasks without mingling the concerns of task execution and task definition.

```plaintext
    multitask-context-dependent-behavior/
    ├── README.md                   # Overall description and instructions
    ├── ROADMAP.md                  # Roadmap to track progress (goals, tasks, open questions)
    ├── mtcdb.code-workspace        # VS Code workspace settings
    ├── meta.env                    # Workspace metadata (environment variables)
    ├── paths.env                   # Central path manager (environment variables)
    ├── setup/                      # Setup scripts and utilities
    │   ├── init.sh                 # Initialization script for Conda environment
    │   ├── post_activate.sh        # Post-activation script for Conda environment
    │   ├── environment.yml         # Conda environment configuration
    │   ├── python.pth              # Paths to Python packages to add to PYTHONPATH (editable mode)
    │   └── bin.pth                 # Paths to binary directories to add to system PATH
    ├── config/                     # Configuration files
    │   ├── dictionaries/           # Dictionaries for spell checking
    │   ├── credentials/            # Credentials for servers
    │   └── tools/                  # Settings for tools and extensions
    ├── src/                        # Source code for Logic/Functionalities/"How" (imported)
    │   ├── core/                   # Main package for analysis, modeling, visualization
    │   ├── ingest/                 # Data ingestion and preprocessing (to perform on the remote hub)
    │   ├── tasks/                  # Administration tasks
    │   │   ├── network/            # Networking tasks (connections, deployment, transfer...)
    │   │   └── ...
    │   └── utils/                  # Helper utilities
    │       ├── io/                 # Input/output functionalities, path management
    │       └── misc/               # Miscellaneous (handling data structures, collections...)
    ├── ops/                        # Entry points for Operations/Execution/"What"-"When"
    │   ├── analysis/               # (organized by types of tasks)
    │   │   ├── preprocess.sh
    │   │   ├── validate.py
    │   │   ├── model.py
    │   │   └── ...
    │   ├── testing/
    │   │   ├── run.sh
    │   │   └── ...
    │   ├── documentation/
    │   │   ├── build.sh
    │   │   └── ...
    │   ├── transfer/
    │   │   ├── connect.sh
    │   │   ├── deploy.sh
    │   │   ├── fetch.sh
    │   │   └── ...
    │   └── maintenance/
    │       ├── inspect.sh
    │       ├── clean.sh
    │       ├── update.sh
    │       └── ...
    ├── tests/                      # Unit tests
    │   └── ...                     # (mirror the structure of the `src/` directory)
    ├── docs/                       # Documentation
    │   ├── build/                  # Output files
    │   ├── source/                 # Source files and configuration
    │   └── reports/                # Reports and summaries
    ├── notebooks/                  # Jupyter notebooks for exploration and visualization
    │   └── ...
    ├── data/                       # Datasets (input and output)
    │   └── ...
    ├── .git/                       # Git workspace
    ├── .gitignore                  # Git ignore file
    ├── .github/                    # GitHub settings and workflows
    ├── archive/                    # Old files kept for reference
    └── ...
```

### Data Storage

```plaintext
    data/
    ├── samples/                    # Sample data for tests and examples
    ├── raw/                        # Raw data (immutable)
    │   ├── ath011b-c1/             # Data from one unit (neuron)
    │   └── ...
    ├── meta/                       # Metadata about experimental events, trials, units
    ├── interim/                    # Intermediate data which has been transformed
    └── processed/                  # Final data sets for modeling
```

## License

This project is licensed under the [GPL License](LICENSE).
