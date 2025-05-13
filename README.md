# Multi-Task Context-Dependent Decision Making - ETL

## Overview

This `data-etl` branch is part of the Multi-Task Context-Dependent Decision Making project,
dedicated to the extraction, transformation, and loading (ETL) stages, prior to analyses.

Those actions operate on secure data servers owned by the laboratory's department.

> [!IMPORTANT]
> This branch is maintained separately from the `main` branch of the project
> repository, to isolate the ETL code so that it can be downloaded and run independently of the main
> project. It is not meant to be merged into the main branch.

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

## License

This project is licensed under the [GPL License](LICENSE).
