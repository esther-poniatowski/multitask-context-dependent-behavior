# Multi-Task Context-Dependent Decision Making - ETL

## Overview

Extract, transform, and load (ETL) data from various sources into a structured format for subsequent
analysis in the Multi-Task Context-Dependent Decision Making project.

This `data-etl` branch is part of a **component-based workflow** in the Multi-Task Context-Dependent
Decision Making project.
It contains all the scripts and configurations necessary to run the very first stage of the project
related to the ETL process of the raw data.
It operates on a secure data server owned by the laboratory's department.
The output of this component serves as the input for subsequent analyses performed by the main
component in the project. 

> [!IMPORTANT]
> This branch is developed and maintained on a separate branch of the project repository. It is not
> meant to be merged into the main branch. Instead, it serves to isolate the ETL code so that it can
> be downloaded and run independently of the main project.


## Workflow

- **Extraction**: Collects raw data from diverse sources (Maryland servers, personal transmission
  channels).
- **Organization**: Arranges files in a logical directory tree.
- **Transformation**: Converts relevant content from raw formats (MATLAB: `.m`, `.mat`, or
  compressed archives: `.zip`, `.tar.gz`) to universal formats (`.csv`), that are compatible with
  Python.
- **Storage**: Saves the formatted data locally for subsequent transfer to a different server were
  downstream analysis will be performed (main component of the project).


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
