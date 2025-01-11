# Archive Log

## Archive Details

- Date of archival: 2025-01-11
- Last commit hash on `main`: 
  - For `setup/`: 3b79343c119c7e44b8d6d8eb14585958164c7fbd
  - For `meta.env`, `path.env`, `NOTES.md`: 6576e357694648379cf5240698e8365e83a27bce

## Reason for Archival

Simplification of the setup process on the main branch: the new set up process only includes a few
steps that can be performed manually and are documented in the `README.md` file. 

The contents of the archive serve for reference if a similar implementation is needed for a more
complex setup process in the future.

## Contents

`setup/`
- Original Location: `./setup/`
- Notes: The `setup` directory contained scripts and configurations that were used to initialize the
  project workspace by running automated bash and conda commands. 

`setup/init.sh`
- Notes: Script to initialize a Conda environment for the workspace (run once).
- Tasks:
  - Create a new Conda environment with the required dependencies from the `environment.yml` file.
  - Symlink the `post_activate.sh` script to the `activate.d` directory of the Conda environment.
  - Symlink the root of the project in the `activate.d` directory of the Conda environment (used int
    the post-activation script).

`setup/post_activate.sh`
- Notes: Script run automatically each time the Conda environment is activated. It was
  symlinked in the Conda environment's `activate.d` directory by the `init.sh` script. 
- Tasks:
  - Navigate to the root directory of the project.
  - Set up global environment variables for the paths to the key directories in the project and the
    project's metadata (name...).
  - Set path variables using the `setup/python.pth` and `setup/bin.pth` files.

`setup/python.pth`
- Notes: List of paths to add paths to the Python path environment variable (`PYTHONPATH`) to enable
  importing modules from the project directory (`src/`, `tests/`...). It uses environment variables
  to the project's key directories.

`setup/bin.pth`
- Notes: Path to the `ops` directory containing scripts to add to the system's binary path
  environment variable (`PATH`) to enable executing custom scripts from the project directory.
