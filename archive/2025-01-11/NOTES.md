# Multi-Task Context-Dependent Decision Making - Programming Notes and Design Choices

## Environment Variables and Paths

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


## Workspace Setup

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


## Programming Languages

Initialization steps are performed in bash since they involve low-level operations (e.g. running
conda commands, sourcing environment variables, modifying path variables).

Configurations that require higher-level operations are performed in Python. Those tasks require
that the conda environment is fully configured and activated.

