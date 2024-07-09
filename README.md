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
Two possible strategies :
- "Inclusion" strategy : Include only the relevant directories.
- "Exclusion" strategy : Include the whole project and exclude irrelevant directories.
Implementation choice : Exclusion.
Advantages :
- When working on any file in VS Code, under the hood, the extensions are run with the
  current file passed as argument (and the additional configuration files). Therefore,
  exclusion should be forced on the irrelevant directories so that they can be worked on.
Warning: Any new directory to exclude must be explicitly added in multiple configuration files.
Before, the relevant directories "src" and "tests" were passed to the extensions,
but now I pass the whole project and exclude
