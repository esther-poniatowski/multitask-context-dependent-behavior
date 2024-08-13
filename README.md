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


## Programming Notes and Implementation Choices

### Environment Variables and Paths

The `.env` file acts as a central path manager. It stores path aliases for other files and scripts
which are imported and run across the project. Each file is referenced by a unique identifier, which
becomes an environment variable. Thereby, if the location of one file is modified, it is only
necessary to update the `.env` file (rather than all the locations where this file is used).

The root of the workspace is determined dynamically when the `.env` file is sourced, using the path
to the `.env` file itself. This ensures the portability of the workspace without having to update
the `.env` file. Importantly, it requires that the `.env` file is *sourced* to enable variable
substitution in the shell environment.

Upon the activation of the conda environment, all the paths in the `.env` file are automatically
available from any other file, since the `.env` is sourced by the post-activation script.


### Workspace Setup

Initializing the workspace's environment involves two phases on distinct time scales :

|              | Initial Setup                            | Post-Activation Setup                  |
|--------------|------------------------------------------|----------------------------------------|
| Script       | `init.sh`                                | `post_activate.sh`                     |
| Tasks        | - Create/update conda environment        | - Set environment variables            |
|              | - Create a symlink to `post_activate.sh` | - Register packages in `PYTHONPATH`    |
|              |   in the `activate.d` directory          | - Register binaries (`ops/`) in `PATH` |
| Activation   | Manual, to create/update the             | Automatic, upon activation of the      |
|              | conda environment                        | conda environment                      |
| Dependencies | - `environment.yml`                      | - `.env`                               |
|              | - (utilities)                            | - `python.pth`                         |
|              |                                          | - `bin.pth`                            |
|              |                                          | - (utilities)                          |


In order to isolate the setup process from the other purposes, the setup files are gathered in a
dedicated `setup/` directory (rather than in the `ops/` directory).

Two types of resources are imported in the setup scripts :

- Configuration files, which are static lists of paths or dependencies (e.g. `environment.yml`,
  `python.pth`, `bin.pth`).
- Utility scripts, which are dynamic and perform operations on the environment (e.g. in `src/`).

Thereby, the setup scripts can focus on the setup workflow rather than on the implementation
details.

The paths to those external resources are specified in the root `.env` file, which has to be
sourced at the *beginning* of each setup script.
To ensure the `.env` file path is correctly resolved in the setup scripts:

- The `init.sh` script has to be run from the *root* directory of the workspace. It creates a
  symlink to the root directory of the workspace in the `activate.d` directory of the conda
  environment.
- The `post_activate.sh` script can be run from any location. It follows the symlink to navigate to
  the root directory of the workspace automatically and to export the `ROOT` environment variable.

In the  it can imported directly since this setup script is run from the root.
In the `post_activate.sh` script, the path to the `.env` file is determined dynamically, since the
script is run from the `activate.d` directory. The path to the real post-activation script is used
as the reference to locate the `.env` file through relative paths (leveraging the simple directory structure between them).


### Programming Languages

Initialization steps are performed in bash since they involve low-level operations (e.g. running
conda commands, sourcing environment variables, modifying path variables).

Configurations that require higher-level operations are performed in Python. Those tasks require
that the conda environment is fully configured and activated.


## Tools and Extensions

Settings for distinct tools are specified in *individual configuration files* (instead of being
stored in a single `pyproject.toml` file or in `.vscode/settings.json`).

Advantages:

- Compatibility : Individual files use the conventional format for the respective tool.
- Readability   : Tools used in the workspace are clearly identified.
- Modularity    : Configurations can be passed independently to different tools.
- Consistency   : Identical configurations are available for tools outside and inside VS Code.

Commands calling those tools have to pass the appropriate configuration file.

### MyPy

Configuration file: `config/tools/mypy.ini`

Command:
```bash
$ mypy --config-file=config/tools/mypy.ini src/ tests/
```

Specific configurations for different parts of the workspace are specified in the same file,
leveraging different sections:

- `[mypy]` : Default configuration for the whole workspace.
- `[mypy-src]` : Configuration for the `src/` directory.
- `[mypy-tests]` : Configuration for the `tests/` directory.

.. warning::
    The `exclude` option in the configuration file only affects recursive directory discovery. When
    calling mypy and explicitly passing a path, it will be checked even if it matches an
    exclusion pattern.
    Idea: Develop a custom plugin that intercepts file processing and skips excluded directories.


### Black

Configuration file: `config/tools/black.toml`
Note: This choice differs from the conventional `pyproject.toml` file, which is used by default.
Here, the goal is to explicitly mark the configuration file for Black by the file name.

Command:
```bash
$ black --config=config/tools/black.toml src/ tests/
```

Exclusions are specified in the `force-exclude` setting.

### Pylint

Configuration file: `config/tools/pylint.ini`
Note: This choice differs from the conventional `.pylintrc` file, which is used by default.
Here, the goal is to prevent hidden files starting with a dot while remaining consistent with the
actual format of the `.pylintrc` file (INI format).

Command:
```bash
$ pylint --rcfile=config/tools/pylint.ini src/ tests/
```

Specific configurations for different parts of the workspace are obtained through a Pylint plugin to
dynamically adjust configurations. This plugin is loaded by specifying the `load-plugins` option in
the `[MASTER]` s
This hook is used to call a function that returns the path to the configuration file based on the
path of the file being linted.

General .pylintrc File: This file specifies the plugin to be loaded.
Plugin Script: The plugin script defines a load_configuration function that modifies Pylint's configuration based on the current working directory.

### Pyright

Configuration file: `config/tools/pyright.json`

Command:
```bash
$ pyright --project config/tools/pyright.json
```

Specific configurations for different parts of the workspace are specified in the
`executionEnvironments` section.

### VS Code Extensions

#TODO

For code analyzers and formatters, include only directories containing codes ("src", "tests")
and exclude all others (docs, notebooks, data, stubs...).
Specify path inclusions/exclusions in configuration files and pass them to the extensions.
Several strategies can be considered :
1. "Inclusion" strategy : Include only the relevant directories.
Disadvantage : If a file path is passed on the command line after the configuration file,
it is included even if it does not belong to the relevant directories.
2. "Exclusion" strategy : Include the whole workspace and exclude irrelevant directories (force).
Advantages : If a file path is passed on the command line after the configuration file,
it is excluded if it belongs to the irrelevant directories.
3. Passing the relevant directories as arguments to the extensions.
Disadvantage : If a file path is passed on the command line after the configuration file,
it leads to an error because it is added after the directories.

Those distinct behaviors are determinant when using "format on save"in VS Code.
Under the hood, the extensions are run with the current file passed as argument (`--stdin-filename`).
Therefore, exclusions should be forced on the irrelevant directories so that they can be opened
in the editor without being formatted.
Warning: Any new directory to exclude must be explicitly added in multiple configuration files.
Before, the relevant directories "src" and "tests" were passed to the extensions,
but now I pass the whole workspace and exclude
Conclusion : Use the "exclusion" strategy.
