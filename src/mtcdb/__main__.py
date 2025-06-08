"""
Command-line entry point for the `mtcdb` package.

Usage
-----
To invoke the package::

    python -m mtcdb


See Also
--------
mtcdb.cli: Command-line interface module for the package.
"""
from .cli import app

app()
