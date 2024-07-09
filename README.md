# Multi-Task Context-Dependent Decision Making

See the [documentation](URL_to_Sphinx_documentation).

## Programming Notes and Implementation Choices

### Pipelines

Keep trace of pipelines used to generate data.

Option 1 : YAML file in the output directory.
Option 2 : Attribute in data themselves.

### Scripts and Notebooks

Scripts : Use to reproduce data in submissions.

Notebooks : Use for tutorials usually.

### Configuration Files

Specify settings in *configuration files* instead of placing them in `.vscode/settings.json`.
Advantage : Configuration files are available for tools *outside of VS Code*,
which allows to run them from the command line (or in Make commands).

Separate settings for different tools in individual configuration files,
instead of storing them in a single `pyproject.toml` file.
Advantages :
- Compatibility : Individual files use the most conventional format.
- Readability : It makes clear which tools are used in the project.
- Modularity : They can be passed independently to different tools.

Create `makefile` commands to pass configuration files to the respective tools.
Those commands should mirror the arguments passed to the VS Code extensions.

### VS Code Extensions

Enable the restructuredText extension even in source codes,
since python files actually use `rst` in docstrings.

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
