# Multi-Task Context-Dependent Decision Making - Programming Notes and Design Choices

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
