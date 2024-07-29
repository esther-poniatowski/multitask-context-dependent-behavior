# Multi-Task Context-Dependent Decision Making

[Documentation](https://esther-poniatowski.github.io/multitask-context-dependent-behavior/).

## Setup

### Environment Variables and Paths

The `.env` file acts as a central path manager. It stores path aliases for other files and scripts
which are used in Makefiles. Each file is referenced by a unique identifier, which becomes an
environment variable. Thereby, if the location of one file is modifies, it is only necessary
to update the `.env` file (and not all the locations where this file is used).

In the main Makefile, it is only necessary to define the path to the `.env` file. This is the only
path which is needed before the environment variables are set in the environment.
Then, all the variables in the `.env` file are automatically available as soon as the environment is
active.

### Tasks Execution

Steps to setup a full environment:

1. Create a `conda` environment using `environment.yml`.
2. Export environment variables from the `.env` file.
3. Register packages in editable mode (source directory, tests...).

The first step is performed directly from bash commands.

Python scripts are used to perform s
Then, *after* the conda environment is created, setup tasks can be performed by python scripts for
simplicity. Each time a python script is used, it is necessary to *activate* the environment before
to ensure that the required packages are available.


## Programming Notes and Implementation Choices

### Configuration Files

Separate settings for different tools in individual configuration files instead of storing them in a
single `pyproject.toml` file.

Advantages :

- Compatibility : Individual files use the most conventional format.
- Readability : It makes clear which tools are used in the project.
- Modularity : They can be passed independently to different tools.

Create `makefile` commands to pass configuration files to the respective tools. Those commands
should mirror the arguments passed to the VS Code extensions. Specify settings in *configuration
files* instead of placing them in `.vscode/settings.json`. Advantage : Configuration files are
available for tools *outside of VS Code*, which allows to run them from the command line (or in Make
commands).

### VS Code Extensions

For code analyzers and formatters, include only directories containing codes ("src", "tests")
and exclude all others (docs, notebooks, data, stubs...).
Specify path inclusions/exclusions in configuration files and pass them to the extensions.
Several strategies can be considered :
1. "Inclusion" strategy : Include only the relevant directories.
Disadvantage : If a file path is passed on the command line after the configuration file,
it is included even if it does not belong to the relevant directories.
2. "Exclusion" strategy : Include the whole project and exclude irrelevant directories (force).
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
but now I pass the whole project and exclude
Conclusion : Use the "exclusion" strategy.
