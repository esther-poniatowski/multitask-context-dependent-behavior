"""
`utils.io_data.saver_base` [module]

Common interface to save data to files.

Classes
-------
`Saver` (ABC)

Implementation
--------------
Separation of Path and Data in Saver Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The `Saver` classes separate the specification of the path and the data to be saved:

- Path: Provided during instantiation of the saver.
- Data: Passed as an argument to the `save` method.

This design offers several advantages:

- Consistency: It aligns the interfaces of the `Saver` and `Loader` classes, promoting uniformity
  across the I/O operations (enforced by the base class `IOHandler`).
- Decoupling via dependency injection : The Saver can be instantiated in the client code which
  specifies the path, before data computation. Then, it can be passed to utility classes that save
  the before data computation, and then passed to utility classes that can invoke the `save` method
  to save the results of their computations, without knowledge of the specific path.

Template Method Design Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The base `save` method defines the sequence of steps in the save operation:

- It calls the abstract `_save` method that is implemented by each subclass, which performs the
  actual saving logic adapted to the file format and data type.
- It catches exceptions and prints a message, which is common to all subclasses.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Union

from utils.io_data.base_io import IOHandler


class Saver(IOHandler):
    """
    Save data to files in an arbitrary format.

    Class Attributes
    ----------------
    EXT : FileExt
        Extension for the specific file format (see `IOHandler.EXT`).

    Attributes
    ----------
    path : Path
        Path to the file where the data will be saved.

    Methods
    -------
    `save`
    `_save` (abstract)

    Raises
    ------
    FileNotFoundError
        If the directory in which to save does not exist.

    Examples
    --------

    See Also
    --------
    `utils.io_data.formats.FileExt`: File extensions.
    `utils.path_system.manage_local.LocalServer`: Utility class.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)  # call the constructor of IOHandler
        if not self.server.check_parent(self.path):
            raise FileNotFoundError(f"Inexistent directory: {self.path.parent}")

    def save(self, data: Any) -> None:
        """
        Save data to a file.

        Arguments
        ---------
        data : Any
            Data to save.

        See Also
        --------
        `utils.path_system.manage_local.LocalServer.check_parent`
            Check the existence of the parent directory.
        """
        try:
            self._save(data)
        except Exception as exc:
            print(f"'{self.__class__.__name__}' failed for path '{self.path}' ")
            raise exc

    @abstractmethod
    def _save(self, data: Any) -> None:
        """Implement the logic to save data to a file."""
