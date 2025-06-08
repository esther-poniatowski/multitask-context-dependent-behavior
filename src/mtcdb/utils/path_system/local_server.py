"""
:mod:`utils.path_system.manage_local` [module]

Classes
-------
:class:`LocalServer`
"""
import os
from pathlib import Path
from typing import Union, Optional

from utils.path_system.base_path_manager import ServerInterface


class LocalServer(ServerInterface):
    """
    Manage the file system of the local server.

    Attributes
    ----------
    root_path : Path
        See :attr:`ServerInterface.root_path`.

    Methods
    -------
    :meth:`enforce_ext`
    :meth:`display_tree`

    See Also
    --------
    For other methods, see the corresponding methods in the :class:`ServerInterface` class.
    """

    def __init__(self, root_path: Optional[Union[str, Path]] = None):
        if root_path is None:  # if no custom path, set using get_root()
            root_path = self.get_root()
        super().__init__(root_path=root_path)  # if still None: `ServerInterface.default_root`

    def get_root(self) -> Union[Path, None]:
        """
        Get the root path of the workspace on the server if the `ROOT` environment variable is set.

        Returns
        -------
        Path, None
            Content of the `ROOT` environment variable if set, else None.
        """
        root = os.environ.get("ROOT")
        return Path(root) if root is not None else None

    def is_dir(self, path: Union[str, Path]) -> bool:
        """
        See :meth:`ServerInterface.is_dir`.

        See Also
        --------
        :func:`pathlib.is_dir`
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.is_dir():
            print(f"[WARNING] Missing directory: {path}")
            return False
        else:
            print(f"[VALID] Existing directory: {path}")
            return True

    def is_file(self, path: Union[str, Path]) -> bool:
        """
        See :meth:`ServerInterface.is_file`.

        See Also
        --------
        :func:`pathlib.is_file`
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            print(f"[WARNING] Missing file: {path}")
            return False
        else:
            print(f"[VALID] Existing file: {path}")
            return True

    def create_dir(self, path: Union[str, Path]) -> None:
        """
        See :meth:`ServerInterface.create_dir`.

        See Also
        --------
        :func:`pathlib.mkdir`: Create a directory.
            Parameter `parents` : Create parent directories if needed.
            Parameter `exist_ok` : If the directory already exists, nothing is done.
        """
        if isinstance(path, str):
            path = Path(path)
        exists = self.is_dir(path)
        if not exists:
            path.mkdir(parents=True, exist_ok=True)
            print(f"[SUCCESS] Directory created: {path}")

    def display_tree(self, path: Union[str, Path], level: int = 0, limit: int = 5) -> None:
        """
        Display the tree structure of a directory.

        Parameters
        ----------
        path : str or Path
        level : int, optional
            Current level in the directory tree, used for indentation.
        limit : int, optional
            Maximum number of items to display per directory.

        Implementation
        --------------

        - Display the name of each file or subdirectory in the currently traversed directory, with
          an indentation level depending on its depth in the hierarchy.
        - If a directory contains more items than the specified limit, show an ellipsis (`...`).
        - Call the method recursively on sub-directories until reaching the end of the directory
        hierarchy (i.e when encountering a directory that contains no subdirectories or files).
        """
        if isinstance(path, str):
            path = Path(path)
        items = list(path.iterdir())
        display_items = items[:limit]
        for item in display_items:
            print("    " * level + "|-- " + item.name)
            if item.is_dir():
                self.display_tree(item, level + 1, limit)
        if len(items) > limit:
            print("    " * level + "|-- ...")
