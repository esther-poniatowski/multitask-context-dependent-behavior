"""
Command-line entry point for the `{{ package_name }}` package.

Usage
-----
To invoke the {{ package_name }} package:

    python -m {{ package_name }}


See Also
--------
{{ package_name }}.cli: Command-line interface module for the {{ package_name }} package.
"""
from .cli import app

app()
