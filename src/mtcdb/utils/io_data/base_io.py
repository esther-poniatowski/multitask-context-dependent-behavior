"""
`utils.io_data.base_loader` [module]

Common interface to load data from files.

Classes
-------
`Loader` (ABC, Generic)
"""

from abc import ABC
from pathlib import Path
from typing import Union, Self

from utils.path_system.local_server import LocalServer


class FileExt(str):
    """
    Extensions for file formats, inherited from `str`.

    Class Attributes
    ----------------
    OPTIONS : FrozenSet[str]
        Valid extension values (strings).

    Methods
    -------
    `is_valid`
    `add_period`
    """

    OPTIONS = frozenset({"csv", "npy", "pkl", "yml"})

    def __new__(cls, ext: str) -> Self:
        ext = cls.add_period(ext)
        if not cls.is_valid(ext):
            raise ValueError(f"Invalid file extension: {ext} not in {cls.OPTIONS}")
        return super().__new__(cls, ext)

    @classmethod
    def is_valid(cls, ext: str) -> bool:
        """Check if a string is a valid extension."""
        return ext in cls.OPTIONS

    @staticmethod
    def add_period(ext: str) -> str:
        """Add a period to the extension if missing."""
        if not ext.startswith("."):
            ext = "." + ext
        return ext


class IOHandler(ABC):
    """
    Base class for loading or saving data from/to files.

    Attributes
    ----------
    EXT : FileExt
        File extension for the specific format to load or save.
    server : LocalServer
        Utility to manage the local file system. It is used to check the existence of the file and
        to enforce the correct file extension.

    Methods
    -------
    `enforce_ext`

    See Also
    --------
    `utils.path_system.manage_local.LocalServer`: Utility class used to interact with the local file
    system (here to check the existence of paths).
    `utils.io_data.base_io.FileExt`: File extensions.
    """

    EXT: FileExt

    def __init__(self, path: Union[str, Path]):
        self.server = LocalServer()
        if isinstance(path, str):
            path = Path(path)
        self.path = self.enforce_ext(path, self.EXT)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}> Path: {self.path}"

    @staticmethod
    def enforce_ext(path: Union[str, Path], ext: Union[str, FileExt]) -> Path:
        """
        Enforce a specific file extension on a path.

        If the file extension is missing or incorrect, it is added or corrected.

        Parameters
        ----------
        path : str or Path
        ext : str or FileExt

        Returns
        -------
        Path
            Path with the correct file extension.

        Raises
        ------
        ValueError
            If the extension does not start with a period.

        See Also
        --------
        `pathlib.Path.with_suffix`
            If the path already contains an extension, it is replaced.
            Otherwise, it is added.
        """
        if isinstance(path, str):
            path = Path(path)
        if isinstance(ext, str):
            ext = FileExt(ext)
        return path.with_suffix(ext)
