# Multi-Task Context-Dependent Decision Making

[Documentation](https://esther-poniatowski.github.io/multitask-context-dependent-behavior/).

## Directory Structure

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

This structure separates the functionality/logic ("how") from the execution/task runners ("what" and
"when"). For chore tasks, if these are complex or involve multiple steps, they can be encapsulated
in separate modules/classes within the `tasks/` directory. This way, the `ops/` scripts can import
and execute these tasks without mingling the concerns of task execution and task definition.


## Data

### Storage

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

### Data types

Raw data consists of the following types:

- `.spk.mat`: Spiking times (in seconds) of one unit in one recording session (MATLAB format).
- `.csv`: Idem in CSV format.
- `.m`: Metadata about a recording session (MATLAB format).


## Initializing the Workspace

1. Clone the repository into a local directory:

```bash
$ git clone git@github.com:esther-poniatowski/multitask-context-dependent-behavior.git
```

2. Navigate to the root directory of the workspace:

```bash
$ cd path/to/multitask-context-dependent-behavior
```

Example paths :

- `cd ~/Documents/Projects/multitask-context-dependent-behavior`
- `cd /Users/eresther/Documents/Work/multitask-context-dependent-behavior`

3. Run the setup script:

```bash
$ bash setup/init.sh
```
