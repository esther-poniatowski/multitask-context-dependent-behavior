# Multi-Task Context-Dependent Decision Making - ETL

## Overview

This `data-etl` branch is part of the Multi-Task Context-Dependent Decision Making project,
dedicated to the extraction, transformation, and loading (ETL) stages, prior to analyses.

Those actions operate on a secure data server owned by the laboratory's department. 

> [!IMPORTANT]
> This branch is developed and maintained on a separate branch of the project repository. It is not
> meant to be merged into the main branch. Instead, it serves to isolate the ETL code so that it can
> be downloaded and run independently of the main project.


## Workflow

Those early stages aim to:

- Collect data from diverse sources and collaborators (Maryland servers, personal transmission
  channels) and organize it on a central data server.
- Organize files in a logical directory tree appropriate for the subsequent analyses.
- Unpack specific content from raw files (MATLAB: `.m`, `.mat`, or compressed archives: `.zip`,
  `.tar.gz`) and export it to universal formats (`.csv`), that are compatible with Python.
- Transfer data to a different server were downstream analysis will be performed (main component of
  the project).


## Installation

1. Connect to the secure data server using SSH.

2. Clone the `data-etl` branch of project's repository:
```bash
git clone --single-branch --branch data-etl https://github.com/esther-poniatowski/multitask-context-dependent-behavior.git
```

3. Install dependencies for MATLAB:

TODO: Add instructions.


## Usage

...


## Directory Structure

Raw and processed files are stored in the following directory structure:
```
/processed_data/
├── site1/
│   ├── session1.csv
│   ├── session2.csv
│   └── ...
├── site2/
│   ├── session1.csv
│   ├── session2.csv
│   └── ...
└── ...
```


## License

This project is licensed under the [MIT License](LICENSE).
