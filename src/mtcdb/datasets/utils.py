#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.dataset.utils` [module]

Base and mixin classes for general functionalities common to specific dataset classes.

Classes
-------
DataHandler
    Handle data files (load, save).

"""
import numpy as np
import os
from typing import Any

from mtcdb.constants import DATA_PATH

class DataHandler:
    """
    Handle data files.

    Attributes
    ----------
    """
    data_dir: str = DATA_PATH
    """Base directory for data files (class attribute)."""
    
    def load(self) -> Any:
        """
        Load data from a file in a specific format.
        
        Note
        ----
        The file path is specified by ``self.data_path``.
        It is loaded using the loader method specified by `self.data_loader``.

        Returns
        -------
        data: Any
            Data loaded from the file.
        
        Exceptions
        ----------
        FileNotFoundError
             If the file is not found, empty data is created for consistency,
            with the right dimensions specified in ``self.empty_shape``.
        
        Raises
        ------
        NotImplementedError
            If the subclass does not define the required attributes, 
            ``data_path``, ``data_loader``, and ``empty_shape``.
        """
        if not hasattr(self, 'data_path'):
            raise NotImplementedError("Missing attribtue : 'data_path'")
        if not hasattr(self, 'data_loader'):
            raise NotImplementedError("Missing attribtue : 'data_loader'")
        if not hasattr(self, 'empty_shape'):
            raise NotImplementedError("Missing attribtue : 'empty_shape'")
        try:
            data = self.data_loader(self.data_path)
        except FileNotFoundError:
            print(f"Data file not found in {self.data_path}")
            print(f"Empty data structure of shape {self.empty_shape}")
            data = np.empty(self.empty_shape)
        return data
    
    def save(self, data: Any) -> None:
        """
        Save data to a file in a specific format.
        
        Note
        ----
        The file path is specified by ``self.data_path``.
        It is saved using the saver method specified by ``self.data_saver``.

        Parameters
        ----------
        data : Any
            Data to be saved to the file.

        Raises
        ------
        NotImplementedError
            If the subclass does not define the required attributes, 
            ``data_path`` or ``data_saver``.
        FileNotFoundError
            If the directory specified in `data_path` does not exist.
        """
        if not hasattr(self, 'data_path'):
            raise NotImplementedError("Missing attribtue : 'data_path'")
        if not hasattr(self, 'data_saver'):
            raise NotImplementedError("Missing attribtue : 'data_saver'")
        directory = os.path.dirname(self.data_path)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Inexistant directory : {directory}.")
        self.data_saver(self.data_path, data)
        print(f"Data saved to {self.data_path}")

