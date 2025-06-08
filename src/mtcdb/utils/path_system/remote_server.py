"""
:mod:`manage_remote` [module]

Classes
-------
:class:`RemoteServer`
"""
import subprocess
from pathlib import Path
from typing import Union, Optional
from dotenv import dotenv_values

from utils.path_system.base_path_manager import ServerInterface


class RemoteServer(ServerInterface):
    """
    Manage the file system of a remote server.

    Class Attributes
    ----------------
    remote_cred : Dict[str, str]
        Mapping of the class attributes to the corresponding environment variables in the `.env`
        file containing the remote server's credentials.
    attr_types : Dict[str, type]
        Mapping of the class attributes to their respective types.

    Attributes
    ----------
    user : str
        Username on the remote server.
    host : str
        IP address or hostname of the remote server.
    root_path : Path
        See :attr:`ServerInterface.root_path`.

    Methods
    -------
    :meth:`load_network_config`

    See Also
    --------
    For other methods, see the corresponding methods in the :class:`ServerInterface` class.
    """

    remote_cred = {"user": "USER", "host": "HOST", "root_path": "ROOT"}
    attr_types = {"user": str, "host": str, "root_path": Path}

    def __init__(
        self,
        user: Optional[str] = None,
        host: Optional[str] = None,
        root_path: Optional[Union[Path, str]] = None,
    ):
        super().__init__(root_path=root_path)
        self.user = user
        self.host = host

    def load_network_config(self, path: Union[Path, str]) -> None:
        """
        Extract user, host, and root path attributes from a `.env` file.

        Arguments
        ---------
        path : Union[Path, str]
            Path to the `.env` file containing the network settings.

        Raises
        ------
        FileNotFoundError
            If the `.env` file is not found at the specified path.
        ValueError
            If the `.env` file does not contain the required network settings.

        See Also
        --------
        :func:`dotenv.dotenv_values`: Store the content of an `.env` file into a dictionary.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] Missing .env file at: {path}")
        env_content = dotenv_values(path)
        print("[SUCCESS] Loaded network settings from .env file at:", path)
        connection_settings = {attr: env_content[var] for attr, var in self.remote_cred.items()}
        for key, value in connection_settings.items():
            if not value:
                raise ValueError(f"[ERROR] Missing network setting in .env file: {key}")
            setattr(self, key, self.attr_types[key](value))
        print(f"[INFO] Credentials for the remote server: {self.user}@{self.host}")
        print(f"       Root path: {self.root_path}")

    def is_dir(self, path: Union[Path, str]) -> bool:
        """
        See :meth:`ServerInterface.is_dir`.

        See Also
        --------
        :command:`test` with p-option `-d`: Check if the path is a directory.
        """
        command = ["ssh", f"{self.user}@{self.host}", f"test -d {path}"]
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            print(f"[VALID] Existing directory: {path} on {self.host}")
            return True
        else:
            print(f"[WARNING] Missing directory: {path} on {self.host}")
            return False

    def is_file(self, path: Union[Path, str]) -> bool:
        """
        See :meth:`ServerInterface.is_file`.

        See Also
        --------
        :command:`test` with option `-f`: Check if the path is a file.
        """
        command = ["ssh", f"{self.user}@{self.host}", f"test -f {path}"]
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            print(f"[VALID] Existing file: {path} on {self.host}")
            return True
        else:
            print(f"[WARNING] Missing file: {path} on {self.host}")
            return False

    def create_dir(self, path):
        """
        See :meth:`ServerInterface.create_dir`.

        See Also
        --------
        :command:`mkdir` with option `-p`(`--parents`): Create parent directories if needed.
        """
        command = ["ssh", f"{self.user}@{self.host}", f"mkdir -p {path}"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"[ERROR] {result.stderr}")
        else:
            print(f"[SUCCESS] Directory created: {path} on {self.host}")
