#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`manage_env_vars` [module]

Manage environment variables in the *active* Conda environment.

Usage
-----
.. code-block:: bash

    python manage_env_vars.py --method set_all --env-path path/to/.env
    python manage_env_vars.py --method unset --vars VAR1 VAR2
    python manage_env_vars.py --method check

Arguments
---------
--method : str, among {'set_all', 'unset', 'check'}
    Method to execute.
--env-path : str, required for `set_all` method
    Path to the `.env` file where the environment variables are defined.
--vars : str, nargs='+', required for `unset` method
    Names of the environment variables to unset.
"""
import argparse
from pathlib import Path
import subprocess
from typing import Dict, List, Union, Callable
from dotenv import dotenv_values

class EnvVarManager:
    """
    Manage environment variables in the *active* conda environment.

    Attributes
    ----------
    env_vars : Dict[str, Union[str, None]]
        Dictionary storing the environment variables loaded from the `.env` file.

    Methods
    -------
    :meth:`load`
    :meth:`set`
    :meth:`set_all`
    :meth:`unset`
    :meth:`check`

    Notes
    -----
    The environment variables configured in the Conda environment are automatically exported to the
    environment when the Conda environment is activated, and removed when it is deactivated.
    Those variables are only available in the *current* shell session, i.e. they are *not
    persistent* across sessions.

    Notes
    -----
    Structure of the ``.env`` file:

    .. code-block:: env

            VAR1=value1
            VAR2=value2

    Structure of the output dictionary:

    .. code-block:: python

            {
                'VAR1': 'value1',
                'VAR2': 'value2',
            }

    See Also
    --------
    :func:`subprocess.run`
        Execute a command via the shell.
        Here, `stdout` and `stderr` are redirected to `subprocess.DEVNULL` to avoid printing the
        output of the conda command.
    """

    def __init__(self, path: Union[str, Path, None] = None):
        self.env_vars: Union[None, Dict[str, Union[str, None]]]
        if path:
            self.env_vars = self.load(path)
        else:
            self.env_vars = None

    def load(self, path) -> Dict[str, Union[str, None]]:
        """
        Load environment variables from one ``.env`` file.

        Arguments
        ---------
        path : env_path : Union[str, Path]
            Path to the `.env` file where the environment variables are defined.

        Returns
        -------
        Dict[str, Union[str, None]]
            Dictionary storing the environment variables.

        See Also
        --------
        :func:`dotenv.dotenv_values`
            Load environment variables from a `.env` file to a dictionary.
        """
        env_vars = dotenv_values(path)
        return env_vars

    @staticmethod
    def set(**vars):
        """
        Set one or several environment variable(s) in the ``conda`` environment.

        Arguments
        ---------
        vars : Dict[str, Any]
            Environment variables to set, passed as keyword arguments.

        See Also
        --------
        :command:`conda env config vars set`
            Conda command to set an environment variable in the active environment.
            Usage: `conda env config vars set VAR_NAME=value`
        """
        for key, value in vars.items():
            if value is not None:
                command = f"conda env config vars set {key}={value}"
                subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f'Set {key}={value}')

    def set_all(self, env_path: Union[str, Path]):
        """Set all the environment variables from the `.env` file in the Conda environment.

        Arguments
        ---------
        env_path : See :meth:`load`
        """
        env_vars = self.load(path=env_path)
        self.set(**env_vars)

    @staticmethod
    def unset(*vars: str):
        """
        Unset one or several environment variable(s) previously set in the ``conda`` environment.

        Arguments
        ---------
        vars : str
            Name(s) of the environment variable(s) to unset.

        See Also
        --------
        :command:`conda env config vars unset`
            Conda command to unset an environment variable in the active environment.
            Usage: `conda env config vars unset VAR_NAME`
        """
        for var in vars:
            command = f"conda env config vars unset {var}"
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
            print(f'Unset {var}')

    @staticmethod
    def check():
        """
        Display the current environment variables set in the `conda` environment.

        See Also
        --------
        :command:`conda env config vars list`
            Conda command to list the environment variables set in the active environment.
        """
        command = "conda env config vars list"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print("ENVIRONMENT VARIABLES")
        print(result.stdout)


class MethodSelector:
    """
    Adapter between the environment variable manager and the command line arguments.

    Class Attributes
    ----------------
    methods : Dict[str, str]
        Mapping from the valid options for the `--method` argument to the methods of :class:`EnvVarManager`.
        Keys: Valid choices for the `--method` argument passed to the command line.
        Values: Attribute names of the methods of :class:`EnvVarManager` to execute.

    Attributes
    ----------
    manager : EnvVarManager
        Instance of the environment variable manager class :class:`EnvVarManager`.
    args : argparse.Namespace
        Command line arguments stored in a namespace.
        See :func:`argparse.ArgumentParser.parse_args`.

    Methods
    -------
    execute()
        Execute the method of the manager based on the `--method` argument.
    get_choices() -> list
        Get the valid choices for the `--method` argument.
    _select_method() -> callable
        Select the method to execute based on the `--method` argument.
    _apply_method(method: callable)
        Apply the selected method to the manager.
    """

    methods = {
        'set_all': 'set_all',
        'set': 'set',
        'unset': 'unset',
        'check': 'check'
    }

    def __init__(self, manager, args):
        self.manager = manager
        self.args = args

    def execute(self):
        """Execute the method of the manager based on the `--method` argument."""
        method = self._select_method()
        self._apply_method(method)

    @classmethod
    def get_choices(cls) -> List[str]:
        """
        Get the valid choices for the `--method` argument.

        Returns
        -------
        choices : List[str]
        """
        return list(cls.methods.keys())

    def _select_method(self) -> Callable:
        """
        Select the method to execute based on the `--method` argument stored in :attr:`args.method`.

        Returns
        -------
        method : callable
            Method of :class:`EnvVarManager` to execute.

        Raises
        ------
        ValueError
            If the method is not recognized.
        """
        method_name = self.methods.get(self.args.method)
        if method_name:
            return getattr(self.manager, method_name)  # get the method by attribute name
        else:
            raise ValueError("Invalid method")

    def _apply_method(self, method: Callable):
        """
        Apply the selected method to the manager.

        Arguments
        ---------
        method : callable
            Output of :meth:`_select_method`.

        Raises
        ------
        ValueError
            If no path is provided to a method which requires one.
        """
        if self.args.method == 'set_all':
            if not self.args.env_path:
                raise ValueError(f"Path required for method: {self.args.method}")
            else:
                method(*self.args.env_path)
        elif self.args.method in ['set', 'unset']:
            if not self.args.vars:
                raise ValueError(f"Variable names required for method: {self.args.method}")
            else:
                method(*self.args.vars)
        elif self.args.method == 'check':
            method()


def main():
    """Set or unset environment variables based on the `--method` argument."""
    parser = argparse.ArgumentParser(description="Manage environment variables in a conda environment.")
    parser.add_argument('--method', type=str,
                        choices=MethodSelector.get_choices(),
                        default='check',
                        help='Method to execute.')
    parser.add_argument('--env-path', type=str, required=False,
                        help='Path to the .env file, required for `set_all`.')
    parser.add_argument('--vars', type=str, nargs='+', required=False,
                        help='Names of variables to unset, required for `set`and `unset`.')
    args = parser.parse_args()
    manager = EnvVarManager()
    method_selector = MethodSelector(manager, args)
    method_selector.execute()


if __name__ == "__main__":
    main()
