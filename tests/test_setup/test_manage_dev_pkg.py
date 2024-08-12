#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_manage_dev_pkg` [module]

See Also
--------
:mod:`manage_dev_pkg`: Tested module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from manage_dev_pkg import DevPackageManager, MethodSelector

@pytest.fixture
def mock_site_packages_path():
    with patch('manage_dev_pkg.site.getsitepackages', return_value=['/fake/site-packages']):
        yield

@pytest.fixture
def mock_pth_file(mock_site_packages_path):
    with patch('manage_dev_pkg.Path') as mock_path:
        mock_path.return_value.exists.return_value = True
        yield mock_path

@pytest.fixture
def package_manager(mock_pth_file):
    return DevPackageManager()

def test_check(package_manager, mock_pth_file):
    """
    Tests for :meth:`DevPackageManager.check`.

    Expected Outputs
    ----------------
    Print the paths registered in the `conda.pth` file.
    """
    mock_pth_file.return_value.open.return_value.read.return_value = "path1\npath2"
    package_manager.check()

def test_register(package_manager):
    """
    Tests for :meth:`DevPackageManager.register`.

    Expected Outputs
    ----------------
    Register the provided paths.
    """
    with patch('manage_dev_pkg.subprocess.run') as mock_run:
        package_manager.register('path/to/package1', 'path/to/package2')
        mock_run.assert_any_call('conda-develop path/to/package1', shell=True)
        mock_run.assert_any_call('conda-develop path/to/package2', shell=True)

def test_unregister(package_manager):
    """
    Tests for :meth:`DevPackageManager.unregister`.

    Expected Outputs
    ----------------
    Unregister the provided paths.
    """
    with patch('manage_dev_pkg.subprocess.run') as mock_run:
        package_manager.unregister('path/to/package1', 'path/to/package2')
        mock_run.assert_any_call('conda-develop -u path/to/package1', shell=True)
        mock_run.assert_any_call('conda-develop -u path/to/package2', shell=True)

def test_clear(package_manager, mock_pth_file):
    """
    Tests for :meth:`DevPackageManager.clear`.

    Expected Outputs
    ----------------
    Clear all paths registered in the `conda.pth` file.
    """
    mock_pth_file.return_value.open.return_value.read.return_value = "path1\npath2"
    with patch('manage_dev_pkg.DevPackageManager.unregister') as mock_unregister:
        package_manager.clear()
        mock_unregister.assert_any_call('path1')
        mock_unregister.assert_any_call('path2')

@pytest.fixture
def args():
    parser = argparse.ArgumentParser(description="Manage developing packages in the workspace.")
    parser.add_argument('--method', type=str, choices=MethodSelector.get_choices(), default='check', help='Method to execute.')
    parser.add_argument('--path-pkg', type=str, nargs='+', help='Path(s) to package(s), required for `register` and `unregister`.')
    return parser.parse_args(['--method', 'register', '--path-pkg', 'path/to/package1', 'path/to/package2'])

def test_method_selector_init(package_manager, args):
    """
    Tests for :class:`MethodSelector.__init__`.

    Expected Outputs
    ----------------
    Initialize the MethodSelector with the package manager and args.
    """
    method_selector = MethodSelector(package_manager, args)
    assert method_selector.package_manager == package_manager
    assert method_selector.args == args

def test_method_selector_execute_register(package_manager, args):
    """
    Tests for :meth:`MethodSelector.execute`.

    Expected Outputs
    ----------------
    Execute the `register` method.
    """
    method_selector = MethodSelector(package_manager, args)
    with patch('manage_dev_pkg.DevPackageManager.register') as mock_register:
        method_selector.execute()
        mock_register.assert_any_call('path/to/package1', 'path/to/package2')

def test_method_selector_execute_unregister(package_manager, args):
    """
    Tests for :meth:`MethodSelector.execute`.

    Expected Outputs
    ----------------
    Execute the `unregister` method.
    """
    args.method = 'unregister'
    method_selector = MethodSelector(package_manager, args)
    with patch('manage_dev_pkg.DevPackageManager.unregister') as mock_unregister:
        method_selector.execute()
        mock_unregister.assert_any_call('path/to/package1', 'path/to/package2')

def test_method_selector_execute_clear(package_manager, args):
    """
    Tests for :meth:`MethodSelector.execute`.

    Expected Outputs
    ----------------
    Execute the `clear` method.
    """
    args.method = 'clear'
    method_selector = MethodSelector(package_manager, args)
    with patch('manage_dev_pkg.DevPackageManager.clear') as mock_clear:
        method_selector.execute()
        mock_clear.assert_called_once()

def test_method_selector_execute_check(package_manager, args):
    """
    Tests for :meth:`MethodSelector.execute`.

    Expected Outputs
    ----------------
    Execute the `check` method.
    """
    args.method = 'check'
    method_selector = MethodSelector(package_manager, args)
    with patch('manage_dev_pkg.DevPackageManager.check') as mock_check:
        method_selector.execute()
        mock_check.assert_called_once()
