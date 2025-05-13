#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates `environment.yml` files for different environments, merging the `pyproject.toml` files of
GitHub repositories.

Objective:

A single project might require distinct environment for distinct tasks (e.g., development,
production), each with its own set of dependencies. Some of those dependencies might come from
GitHub repositories, without being distributed on conda channels. To maximize environment
robustness, the final `environment.yml` file of the super-project should include all the
conda-installable dependencies of those external repositories. This requires merging the dependency
specifications of all those external repositories into a single `environment.yml` file per
environment.

Requirements
------------

Each external repository should have a `pyproject.toml` file at its root, specifying the
dependencies in a format compatible with the `unidep` package.

Configuration
-------------
The environment names and the GitHub repositories should be specified in a YAML file, which is
passed as an argument to this script. The YAML file should contain a list of repositories,
structured as follows:

```yaml
- name: repo_name_1
    user: user_name
    branch: main
    pyproject_path: pyproject.toml
    envs:
    - dev
    - prod
- name: repo_name_2
    envs:
    - dev
    - etl-dev
```

Only the `name` and `env` fields are mandatory. The latter contain a list of environment names where
the repository is used.

Usage
-----
To general the merged files for all the environments in a specification file:

```sh
python ./scripts/compose_envs.py --spec ./env_spec.yml
```

To generate specific environment files from the specification file:

```sh
python ./scripts/compose_envs.py --spec ./env_spec.yml --env dev prod
```

To specify the platforms to include in the merged file:

```sh
python ./scripts/compose_envs.py --spec ./env_spec.yml --platforms linux-64 osx-64 win-64
```

Arguments
---------
--spec : str
    Path to the specification file for environments (e.g., `env_spec.yml`).
--env : str, optional
    Type(s) of environments to define (keys in `env_spec.yml`).
    If not specified, all environments are included.
--platforms : str, optional
    Platforms to include in the merged `environment.yml` files (e.g., linux-64 osx-64 win-64).
--output-dir : str, optional
    Directory where the merged `environment.yml` files are saved.
--tmp-dir : str, optional
    Temporary directory where the `pyproject.toml` files are stored.
--user : str, optional
    Default GitHub username from which the repositories are recovered.
--branch : str, optional
    Default branch name to use for the repositories.
--pyproject-path : str, optional
    Default path to the `pyproject.toml` file from the repository roots.
--help : bool
    Show this help message and exit.

See Also
--------
unidep
    Unified Conda and Pip requirements management: https://unidep.readthedocs.io/en/latest/

"""
import argparse
from pathlib import Path
from typing import List, Dict
import shutil

import requests
from pydantic import BaseModel, Field
from unidep._cli import _merge_command
import yaml


# --- Default Variables ----------------------------------------------------------------------------

# Default username for GitHub repositories
USER = 'esther-poniatowski'

# Default branch to use to recover `pyproject.toml` files
BRANCH = 'main'

# Default path to `pyproject.toml` files from repository roots
PYPROJECT_PATH = 'pyproject.toml'

# List of platforms to include in the merged file
PLATFORMS = ('linux-64', 'linux-aarch64', 'linux-ppc64le', 'osx-64', 'osx-arm64', 'win-64')

# Destination directory for downloaded files
TMP_DIR = Path('include/tmp')

# Destination directory for merged files (project root)
OUTPUT_DIR = Path('./')


# --- Utilities ------------------------------------------------------------------------------------

class Repo(BaseModel):
    """
    Represents a GitHub repository and its associated environment(s).

    Attributes
    ----------
    name : str
        Name of the GitHub repository.
    user : str, optional
        GitHub username.
    branch : str, optional
        Branch name to use.
    pyproject_path : str, optional
        Path to the `pyproject.toml` file from the repository root.
    envs : List[str]
        Environment names where the repository is used.
    """
    name: str
    user: str
    branch: str
    pyproject_path: str
    envs: List[str] = Field(default_factory=list)


def load_env_spec(spec_path: Path,
                  default_user: str = USER,
                  default_branch: str = BRANCH,
                  default_pyproject_path: str = PYPROJECT_PATH,
                  ) -> List[Repo]:
    """
    Loads an environments specification from a YAML file.

    The YAML file should define a list of GitHub repositories, specifying fields corresponding to
    the `Repo` class:

    ```yaml
    - name: repo_name_1
      user: user_name
      branch: main
      pyproject_path: pyproject.toml
      envs:
        - dev
        - prod
    - name: repo_name_2
      envs:
        - dev
        - etl-dev
    ```

    Only the `name` and `env` fields are mandatory. The latter contain a list of environment names
    where the repository is used.

    The resulting dictionary contains a list of `Repo` objects which directly reflect the content of
    the YAML file.

    Arguments
    ---------
    spec_path : Path
        Path to the YAML file containing the environment specification.

    Returns
    -------
    repositories : List[Repo]
        `Repo` objects representing the repositories and their associated environments.

    Raises
    ------
    ValueError
        If the YAML file does not contain a list of repository specifications or if any required
        fields are missing.
    TypeError
        If the YAML file does not contain a list of repository specifications or if any required
        fields are missing.
    OSError
        If an error occurs when reading the YAML file.
    """
    with open(spec_path, 'r', encoding='utf-8') as file:
        raw_data = yaml.safe_load(file) # dictionary
    raw_data = yaml.safe_load(file)
    if not isinstance(raw_data, list):
        raise ValueError("Expected a list of repository specifications.")
    repositories = []
    for repo_info in raw_data:
        if not isinstance(repo_info, dict):
            raise ValueError("Invalid format for repository entry: expected a dictionary.")
        if 'name' not in repo_info:
            raise ValueError(f"Missing 'name' field in repository entry: {repo_info}.")
        if 'envs' not in repo_info:
            raise ValueError(f"Missing 'envs' field in repository '{repo_info['name']}'.")
        repo = Repo(
            name=repo_info['name'],
            user=repo_info.get('user', default_user),
            branch=repo_info.get('branch', default_branch),
            pyproject_path=repo_info.get('pyproject_path', default_pyproject_path),
            envs=repo_info['envs'] # mandatory field
        )
        repositories.append(repo)
    return repositories


def specify_environments(repositories: List[Repo]) -> Dict[str, List[Repo]]:
    """
    Identifies the distinct environments to create and the required repositories to use for each.

    Arguments
    ---------
    repositories : List[Repo]
        All collected repository information.

    Returns
    -------
    environments : Dict[str, List[Repo]]
        Dictionary mapping each environment to its corresponding repositories.
    """
    environments: Dict[str, List[Repo]] = {}
    for repo in repositories:
        for env in repo.envs:
            if env not in environments:
                environments[env] = []
            environments[env].append(repo)
    return environments


def build_url(user:str, repo:str, branch:str, pyproject_path:str) -> str:
    """
    Constructs the URL to the `pyproject.toml` file in a GitHub repository.

    See Also
    --------
    https://docs.github.com/en/rest/reference/repos#get-repository-content
    """
    return f'https://raw.githubusercontent.com/{user}/{repo}/{branch}/{pyproject_path}'


def build_file_path(repo: Repo, tmp_dir: Path = TMP_DIR) -> Path:
    """
    Constructs the file path to save the `pyproject.toml` file in the temporary directory.

    Arguments
    ---------
    repo : Repo
        Repository information.
    tmp_dir : Path, default=TMP_DIR
        Temporary directory for saving downloaded files.

    Returns
    -------
    file_path : Path
        Path to the `pyproject.toml` file in the temporary directory.
    """
    return tmp_dir / repo.name / 'pyproject.toml'


def build_env_file_path(env_name: str, dest_dir: Path = OUTPUT_DIR) -> Path:
    """
    Constructs the file path to save the merged `environment.yml` file in the project.

    Arguments
    ---------
    env_name : str
        Name of the environment.
    dest_dir : Path, default=OUTPUT_DIR
        Destination directory for saving the merged file.

    Returns
    -------
    file_path : Path
        Path to the merged `environment.yml` file.
    """
    return dest_dir / f'environment_{env_name}.yml'


def download_pyproject_toml(repositories: List[Repo], tmp_dir: Path = TMP_DIR, timeout = 10) -> None:
    """
    Downloads the `pyproject.toml` files from the specified GitHub repositories.

    Missing fields in repository information are filled with default values (e.g., user, branch,
    pyproject_path).

    Each file is downloaded to a temporary directory, in a subfolder named after the repository.

    Arguments
    ---------
    repositories : List[Repo]
        Repository information for all the files to download.
    tmp_dir : Path, default=TMP_DIR
        Temporary directory for saving downloaded files.
    timeout : int, default=10
        Timeout for the HTTP request in seconds.

    Raises
    ------
    requests.RequestException
        If an error occurs while sending the HTTP request.
    OSError
        If an error occurs while writing the file to the destination path.
    FileNotFoundError
        If the destination directory does not exist.

    See Also
    --------
    requests
        Python library to send HTTP requests: https://docs.python-requests.org/en/latest/
    """
    # Create the temporary directory for saving if it does not exist
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Download each repository's file from GitHub
    for repo in repositories:
        user = repo.user
        repo_name = repo.name
        branch = repo.branch
        pyproject_path = repo.pyproject_path
        raw_url = build_url(user, repo_name, branch, pyproject_path)
        dest_path = build_file_path(repo, tmp_dir)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Check if the file already exists
            response = requests.get(raw_url, timeout=timeout)
            if response.status_code == 200:
                # Write the content to the destination file
                with open(dest_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {repo}: `pyproject.toml` saved to {dest_path}")
            else:
                print(f"Failed to download {repo}: HTTP {response.status_code}")
        except (requests.RequestException, OSError) as exc:
            print(f"Error downloading {repo}: {exc}")

def merge_dependencies(env_name: str,
                       env_repos: List[Repo],
                       tmp_dir: Path = TMP_DIR,
                       output_dir: Path = OUTPUT_DIR,
                       platforms = PLATFORMS,
                       ) -> None:
    """
    Merges the dependencies from the `pyproject.toml` files of the specified repositories.

    Arguments
    ---------
    env_name : str
        Name of the environment.
    env_repos : List[Repo]
        List of repositories for the environment.
    tmp_dir : Path, default=TMP_DIR
        Temporary directory where the `pyproject.toml` files are stored.
    output_dir : Path, default=OUTPUT_DIR
        Directory where the merged `environment.yml` file are saved.
    platforms : List[str], optional
       Platforms to include in the merged file. If not specified, all platforms are included.
       Common values include: ?

    Notes
    -----
    The `_merge_command` function from the `unidep` package is used to perform the merging. All
    arguments are mandatory, even though some are ignored or useless in this case.
    In the source code, the argument `files` is set to None, and the CLI does not accept the `files`
    option. Here, the `files` argument is set to the list of `pyproject.toml` files to merge, and
    the `directory` and `depth` arguments are ignored.

    See Also
    --------
    unidep._cli._merge_command
        Merge serveral `pyproject.toml` files into a single `environment.yml` file.
        https://github.com/basnijholt/unidep/blob/main/unidep/_cli.py
        https://unidep.readthedocs.io/en/latest/as-a-cli.html
    """
    files = [build_file_path(repo, tmp_dir) for repo in env_repos]
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"{file} does not exist.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _merge_command(
        depth=1,             # ignored since `files` is provided
        directory=tmp_dir,   # ignored since `files` is provided
        files=files,
        name=env_name,
        output=build_env_file_path(env_name, output_dir),
        stdout=False,
        selector="sel",      # default
        platforms=list(platforms),
        ignore_pins=[],
        skip_dependencies=[],
        overwrite_pins=[],
        verbose=False
    )


def clear_tmp_dir(tmp_dir: Path = TMP_DIR) -> None:
    """
    Clears the temporary directory by removing all files and subdirectories.

    Arguments
    ---------
    tmp_dir : Path, default=TMP_DIR
        Temporary directory to clear.

    See Also
    --------
    shutil.rmtree
        Python function to remove a directory tree recursively :
        https://docs.python.org/3/library/shutil.html#shutil.rmtree
    """
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print(f"Temporary directory {tmp_dir} cleared.")


# --- Main Functionality ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Define the command-line arguments
    parser = argparse.ArgumentParser(description='Generate `environment.yml` files for different environments.')
    parser.add_argument('--spec',
                        type=str,
                        required=True,
                        help='Path to the specification file for environments (e.g., `env_spec.yml`).')
    parser.add_argument('--env',
                        type=str,
                        nargs='*',
                        required=False,
                        help='Type(s) of environments to define (keys in `env_spec.yml`).')
    parser.add_argument('--platforms',
                        type=str,
                        nargs='*',
                        default=list(PLATFORMS),
                        help='Platforms to include in the merged `environment.yml` files (e.g., linux-64 osx-64 win-64)')
    parser.add_argument('--output-dir',
                        type=str,
                        default=OUTPUT_DIR,
                        help='Directory where the merged `environment.yml` files are saved.')
    parser.add_argument('--tmp-dir',
                        type=str,
                        default=TMP_DIR,
                        help='Temporary directory where the `pyproject.toml` files are stored.')
    parser.add_argument('--user',
                        type=str,
                        default=USER,
                        help='Default GitHub username from which the repositories are recovered.')
    parser.add_argument('--branch',
                        type=str,
                        default=BRANCH,
                        help='Default branch name to use for the repositories.')
    parser.add_argument('--pyproject-path',
                        type=str,
                        default=PYPROJECT_PATH,
                        help='Default path to the `pyproject.toml` file from the repository roots.')

    # Parse the command-line arguments
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)

    # Load the environment specification from the YAML file
    env_spec = load_env_spec(Path(args.spec),
                             default_user=args.user,
                             default_branch=args.branch,
                             default_pyproject_path=args.pyproject_path)
    envs = specify_environments(env_spec)
    if args.env is None: # select all environments
        envs_to_create = list(envs.keys())
    else: # select specific environments from the specification file
        envs_to_create = args.env
        for env in envs_to_create:
            if env not in envs:
                raise ValueError(f"Invalid environment '{env}' specified. Available environments: {list(envs.keys())}")

    # Download the `pyproject.toml` files from the specified GitHub repositories
    download_pyproject_toml(
        repositories=[repo for env in envs_to_create for repo in envs[env]],
        tmp_dir=tmp_dir
    )

    # Merge the dependencies from the `pyproject.toml` files of the specified repositories
    for env_name in envs_to_create:
        merge_dependencies(
            env_name=env_name,
            env_repos=envs[env_name],
            tmp_dir=tmp_dir,
            output_dir=output_dir,
            platforms=args.platforms
        )

    # Clear the temporary directory
    clear_tmp_dir(tmp_dir=Path(args.tmp_dir))
