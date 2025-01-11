#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PathManager Class

Manage environment variable paths: `PATH`, `PYTHONPATH`, `MATLABPATH`.

Usage
-----
As a class:

.. code-block:: python

    pm = PathManager('PYTHONPATH')
    pm.add('/path/to/directory')
    pm.remove('/path/to/remove')
    print(pm)
    pm.reset()

As a script:

.. code-block:: bash

    python path_manager.py --env_var PYTHONPATH --add /path/to/directory
    python path_manager.py --env_var PYTHONPATH --add_from_file /path/to/file.txt
    python path_manager.py --env_var PYTHONPATH --remove /path/to/remove
    python path_manager.py --env_var PYTHONPATH --show
    python path_manager.py --env_var PYTHONPATH --reset

Notes
-----
Formats:

- Environment Variables: Paths are stored as a single string where individual paths are separated by a platform-specific delimiter. On UNIX-like systems (Linux, macOS), the delimiter is a colon (:).
  Example: `/usr/local/bin:/usr/bin:/bin`
- Internal Representation (self.paths): Paths are stored as a Python list of strings.
  Example: ['/usr/local/bin', '/usr/bin', '/bin']

Conversion Between Formats:

- From Environment Variable to self.paths: Split the string using the platform-specific delimiter (e.g., colon : on UNIX).
- From self.paths to Environment Variable: Joined back the list into a single string using the same delimiter.

"""

import os
import argparse

class PathManager:
    VALID_ENV_VARS = ['PATH', 'PYTHONPATH', 'MATLABPATH']

    def __init__(self, env_var):
        if env_var not in self.VALID_ENV_VARS:
            raise ValueError(f"Invalid environment variable '{env_var}'. Must be one of {self.VALID_ENV_VARS}.")
        self.env_var = env_var
        self.initial_paths = self._get_paths()
        self.paths = self.initial_paths.copy()

    def add(self, path):
        """Add a new path to the environment variable if it is not already present."""
        if path not in self.paths:
            self.paths.append(path)
            self._update_env_var()

    def add_from_file(self, file_path):
        """Add paths from a file to the environment variable."""
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    path = line.strip()
                    if path:
                        self.add(path)
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}.")

    def remove(self, path):
        """Remove a path from the environment variable if it exists."""
        if path in self.paths:
            self.paths.remove(path)
            self._update_env_var()

    def reset(self):
        """Reset the environment variable to its initial state."""
        self.paths = self.initial_paths.copy()
        self._update_env_var()

    def _get_paths(self):
        """Retrieve the current paths from the environment variable."""
        return os.environ.get(self.env_var, "").split(os.pathsep)

    def _update_env_var(self):
        """Output the shell command to update the environment variable."""
        print(f'export {self.env_var}="{os.pathsep.join(self.paths)}"')

    def __repr__(self):
        """Display the current paths."""
        return "\n".join(self._get_paths())

def main():
    parser = argparse.ArgumentParser(description="Manage environment variable paths.")
    parser.add_argument('--env_var', required=True, help="Environment variable to manage.")
    parser.add_argument('--add', help="Add a path to the environment variable.")
    parser.add_argument('--add_from_file', help="Add paths from a file to the environment variable.")
    parser.add_argument('--remove', help="Remove a path from the environment variable.")
    parser.add_argument('--reset', action='store_true', help="Reset the environment variable to its initial state.")
    parser.add_argument('--show', action='store_true', help="Show the current paths in the environment variable.")

    args = parser.parse_args()

    try:
        pm = PathManager(args.env_var)
    except ValueError as e:
        print(e)
        return

    if args.add:
        pm.add(args.add)
        print(f"Added to {args.env_var}: {args.add}")
    elif args.add_from_file:
        pm.add_from_file(args.add_from_file)
        print(f"Added to {args.env_var}: paths from {args.add_from_file}")
    elif args.remove:
        pm.remove(args.remove)
        print(f"Removed from {args.env_var}: {args.remove}")
    elif args.reset:
        pm.reset()
        print(f"Reset {args.env_var} to its initial state.")
    elif args.show:
        print(f"Current paths in {args.env_var}:")
        print(pm)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
