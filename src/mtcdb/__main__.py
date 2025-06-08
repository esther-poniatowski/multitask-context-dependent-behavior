"""
Entry point for the `mtcdb` package, invoked as a module.

Usage
-----
To launch the command-line interface, execute::

    python -m mtcdb


See Also
--------
mtcdb.cli: Module implementing the application's command-line interface.
"""
from .cli import app

app()
