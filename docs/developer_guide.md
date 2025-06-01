# Developer Guide

## Synchronizing Dependencies

This project uses the `unidep` tool to synchronize dependencies between Pip and Conda.

To specify the dependencies in this framework, proceed as follows:

1. Specify all the in the `pyproject.toml` (central source of truth).

    - Core dependencies: under the section `[tool.unidep.dependencies]`.
    - Optional dependencies: under the section `[tool.unidep.optional_dependencies]`.

    Distinguish several dependency groups to allow modular installations (e.g., for development,
    runtime).

    Include personal dependencies via their GitHub URL.

2. Specify the GitHub repositories used as dependencies in an `env_spec.yml` file. See the required
   format in the script `scripts/compose_envs.py` which parses the `env_spec.yml` file.

3. Generate `environment.yml` files for conda by running the script:

    ```sh
    python scripts/compose_envs.py --spec env_spec.yml
    ```

    This will create  `environment.yml` files for each environment in the project, merging all the
    dependencies required by the external personal libraries.
