#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`manage_dev_pkg` [module]

Manage developing packages in the workspace.

Usage
-----
.. code-block:: bash

    python manage_dev_pkg.py --method check
    python manage_dev_pkg.py --method register --path-pkg /path/to/package1 /path/to/package2
    python manage_dev_pkg.py --method unregister --path-pkg /path/to/package2
    python manage_dev_pkg.py --method clear

Arguments
---------
--method : str, among {'check', 'register', 'unregister', 'clear'}
    Method to execute.
--path-pkg : str, required for 'register' and 'unregister' methods
    Path to a directory containing packages register or unregister.

Warning
-------
Activate the conda environment before running this script.
"""
import argparse
from pathlib import Path
import site
import subprocess
from typing import List, Dict, Callable


class DevPackageManager:
    """
    Manage packages in the *active* conda environment.

    Attributes
    ----------
    site_packages : str
        Path to the site-packages directory of the conda environment.
    pth_file : Path
        Path to the conda.pth file.

    Methods
    -------
    :meth:`check`
    :meth:`register`
    :meth:`unregister`

    Notes
    -----
    Packages registered in *development mode* in the conda environment can be imported by any code
    located in the *same environment*. Any change to the source files immediately affects the
    package in the environment without needing a reinstall.

    The registration process merely links a directory containing packages to the virtual
    environment, instead of copying the files or building the package.
    Registered paths are appended to the ``conda.pth`` file in the ``site-packages`` directory.

    .. code-block:: plaintext

        ~/miniconda3/
        └── envs/
            └── myenv/
                └── lib/
                    └── pythonX.Y/
                        └── site-packages/
                            └── conda.pth

    See Also
    --------
    :func:`site.getsitepackages`:
        Get the site-packages directory of the conda environment.
    :func:`subprocess.run`
        Execute shell commands from Python.
    `conda-build <https://docs.conda.io/projects/conda-build/en/latest/>`_
        Required to run the `conda-develop` command.
    """
    def __init__(self):
        self.site_packages = site.getsitepackages()[0]  # first site-packages directory
        self.pth_file = Path(self.site_packages) / "conda.pth"

    def check(self):
        """Identify paths registered for packages in development mode in :attr:`pth_file`."""
        print(f"PACKAGES IN DEVELOPMENT MODE")
        print(f"Registered in: {self.pth_file}")
        if self.pth_file.exists():
            with self.pth_file.open("r") as f:
                for line in f.readlines():
                    print(line.strip())
        else:
            print(f"No packages registered ({self.pth_file} not created).")

    def register(self, *paths: str):
        """
        Register package(s) in development mode by adding path(s) to :attr:`pth_file`.

        Arguments
        ---------
        paths : str
            Path(s) to register, where the package(s) to develop are stored.

        Raises
        ------
        FileNotFoundError
            If any path does not exist.
        """
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Path not found: {path}")
            command = f"conda-develop {path}"
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
            print(f"Registered: {path}")

    def unregister(self, *paths: str):
        """
        Unregister package(s) from development mode by removing path(s) from :attr:`pth_file`.

        Arguments
        ---------
        paths : str
            Path(s) to unregister.
        """
        for path in paths:
            command = f"conda-develop -u {path}"
            subprocess.run(command, shell=True,  stdout=subprocess.DEVNULL)
            print(f"Unregistered: {path}")

    def clear(self):
        """
        Clear all paths previously registered in :attr:`pth_file`.

        This method allows to reinitialize all the registered packages before a new registration.
        This is especially useful when source directories were moved to another location, which has
        broken the link between the developing packages and the environment.

        Parameters
        ----------
        path : str
            Path to unregister.
        """
        if self.pth_file.exists():
            with self.pth_file.open("r") as f:
                paths = f.readlines()
            for path in paths:
                self.unregister(path.strip())  # remove leading/trailing whitespaces
        else:
            print(f"No file at {self.pth_file}")


class MethodSelector:
    """
    Adapter between the package manager and the command line arguments.

    Class Attributes
    ----------------
    methods :  Dict[str, str]
        Mapping from the valid options for the `--method` argument to the methods of :class:`DevPackageManager`.
        Keys: Valid choices for the `--method` argument passed to the command line.
        Values: Attribute names of the methods of :class:`DevPackageManager` to execute.

    Attributes
    ----------
    package_manager : DevPackageManager
        Instance of the package manager class :class:`DevPackageManager`.
    args : argparse.Namespace
        Command line arguments stored in a namespace.
        See :func:`argparse.ArgumentParser.parse_args`.

    Methods
    -------
    :meth:`execute`
    :meth:`get_choices`
    :meth:`_select_method`
    :meth:`_apply_method`
    """

    methods = {
        'check': 'check',
        'register': 'register',
        'unregister': 'unregister',
        'clear': 'clear'
    }

    def __init__(self, package_manager, args):
        self.package_manager = package_manager
        self.args = args

    def execute(self):
        """Execute the method of the package manager based on the `--method` argument."""
        method = self._select_method()
        self._apply_method(method)

    @classmethod
    def get_choices(cls) -> List[str]:
        """
        Get the valid choices for the `--method` argument.

        Returns
        -------
        choices : list of str
        """
        return list(cls.methods.keys())

    def _select_method(self) -> Callable:
        """
        Select the method to execute based on the `--method` argument stored in :attr:`args.method`.

        Returns
        -------
        method : callable
            Method of :class:`DevPackageManager` to execute.

        Raises
        ------
        ValueError
            If the method is not recognized.
        """
        method_name = self.methods.get(self.args.method)
        if method_name:
            return getattr(self.package_manager, method_name)  # get the method by attribute name
        else:
            raise ValueError("Invalid method")

    def _apply_method(self, method: Callable):
        """
        Apply the selected method to the package manager.

        Arguments
        ---------
        method : callable
            Output of :meth:`_select_method`.

        Raises
        ------
        ValueError
            If no path is provided to a method which requires at least one.
        """
        if self.args.method in ['register', 'unregister']:
            if not self.args.path_pkg:
                raise ValueError(f"Path(s) required for method: {self.args.method}")
            else:
                method(*self.args.path_pkg)
        else:
            method()


def main():
    """Execute the package management process based on the `--method` argument."""
    parser = argparse.ArgumentParser(description="Manage developing packages in the workspace.")
    parser.add_argument('--method', type=str,
                        choices=MethodSelector.get_choices(),  # set valid choices
                        default='check',
                        help='Method to execute.')
    parser.add_argument('--path-pkg', type=str, nargs='+',
                        help='Path(s) to package(s), required for `register` and `unregister`.')
    args = parser.parse_args()
    # Select and execute the method
    package_manager = DevPackageManager()
    method_selector = MethodSelector(package_manager, args)
    method_selector.execute()


if __name__ == "__main__":
    main()
