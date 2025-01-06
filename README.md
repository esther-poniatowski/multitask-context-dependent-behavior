# Multi-Task Context-Dependent Decision Making

- [Description](#description)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Description

...

**Full Documentation**: [multitask-context-dependent-behavior](https://esther-poniatowski.github.io/multitask-context-dependent-behavior/)

## Authors

**Esther Poniatowski** | @esther-poniatowski | esther.poniatowski@ens.psl.eu


## Acknowledgments

...


## Installation

> [!TIP]
> **Prerequisites**
> Ensure that the following tools are available on the local machine:
> - git
> - conda

To set up this project on a local machine, follow the steps below:

### Initialize a local copy of the repository

1. Navigate to the local directory where the root folder of the repository should reside.
  
2. Clone the repository:

```bash
$ git clone git@github.com:esther-poniatowski/multitask-context-dependent-behavior.git
```

The repository files are installed into a new directory named `multitask-context-dependent-behavior`.


### Create a virtual environment

1. Create an dedicated conda environment containing all the dependencies:

```
conda env create -f environment.yml
```

The new conda environment is named `mtcdb`. 

2. (Optional) Register the packages in "editable mode":

```
conda activate mtcdb
pip install -e /src/<package-name>
```

## Usage

...


## Contributing

> [!IMPORTANT]
> To contribute effectively, please conform to those guidelines and use the provided templates.

### Configure the workspace (optional, for contributing)

1. Ask the author to share the Personal Access Token of the repository.

2. Navigate to the root directory of the local repository:

```bash
$ cd path/to/multitask-context-dependent-behavior
```

3. Add the authentication user name and token in the `.git/credentials` file:

```bash
https://<username>:<personal-access-token>@github.com
```

4. Configure the credential.helper to use the credentials file:

```bash
git config credential.helper store
```

5. Configure the user profile:

```bash
git config user.name "Example Name"
git config user.email " exampleemail@domain.com"
```

6. Edit the commit message template `.gitmessage` with the corresponding user name and email.


### Sumbitting Issues

To submit a new issue:

1. In the repository page, navigate to the "Issues" tab and click on "New Issue".
2. Select and fill the issue template.
3. Add relevant labels, assignees, and milestone if applicable.

### Using the Commit Message Template

1. Navigate inside the repository directory:
```
cd <repository-name>
```
   
2. Edit the commit template (`.gitmessage`) to specify the author name.

3. Configure `git` to use this file as a commite template:
```
git config commit.template .gitmessage
```
   
4. Verify the configuration:
```
git config --get commit.template
```

> [!NOTE]
> To write a commit message with this template, adhere to the following format:
>
> - Capitalize the subject, do not add a period at the end
> - Limit the subject line to 50 characters
> - Use the imperative mood in the subject line
> - Separate subject from body with a blank line
> - Wrap the body at 72 characters per line
> - Use the body to explain what and why (not how)
> - Add references to issues or other commits using [GitHub keywords](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests)


---


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


