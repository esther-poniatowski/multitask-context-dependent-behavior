#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`utils.io_data.savers.impl_savers` [module]

Save data from files in specific formats.

Any object which needs to save data can interact with one Saver subclass.

Classes
-------
:class:`SaverPKL`
:class:`SaverNPY`
:class:`SaverCSV`

See Also
--------
:class:`utils.io_data.formats.FileExt`: File extensions.
:class:`utils.io_data.savers.base_saver.Saver`: Base class for savers.
"""

import pickle
from types import MappingProxyType
from typing import Any

import csv
import dill
import numpy as np
import pandas as pd

from utils.io_data.formats import FileExt
from utils.io_data.savers.base_saver import Saver


class SaverPKL(Saver):
    """
    Save data in the Pickle format.

    Notes
    -----
    Any object can be saved in a Pickle file, therefore the class attribute :attr:`save_methods` is
    not needed.
    Since it is not possible to enumerate all the types that can be saved in a Pickle file, the
    method :meth:`_save` is overridden to dodge the checking step of the base method.
    """

    ext = FileExt.PKL

    def save(self):
        """
        Save any Python object to a Pickle file.

        Warning
        -------
        Override the base method to avoid the checking step.

        See Also
        --------
        :func:`pickle.dump`
        """
        self.server.check_parent(self.path)
        self.path = self.server.enforce_ext(self.path, self.ext)
        with self.path.open("wb") as file:
            pickle.dump(self.data, file)
        print(f"[SUCCESS] Saved to {self.path}")


class SaverDILL(Saver):
    """
    Save data in the Dill format.

    Notes
    -----
    See :class:`SaverPKL` for explanations.

    Objects handled by the Dill format and not by the Pickle format (examples):

    - Functions and classes.
    - Objects that contain references to functions or classes.
    - Objects that contain circular references.
    - Objects that contain lambda functions, nested functions.
    - Objects that contain NumPy arrays with dtype=object or with numpy.ma module.

    See Also
    --------
    :func:`dill.dump`
    """

    ext = FileExt.PKL

    def save(self):
        """
        Save any Python object to a Dill file.

        Warning
        -------
        Override the base method to avoid the checking step.

        See Also
        --------
        :func:`dill.dump`
        """
        self.server.check_parent(self.path)
        self.path = self.server.enforce_ext(self.path, self.ext)
        with self.path.open("wb") as file:
            dill.dump(self.data, file)
        print(f"[SUCCESS] Saved to {self.path}")


class SaverNPY(Saver):
    """Save data in the NPY format."""

    ext = FileExt.NPY
    save_methods = MappingProxyType({np.ndarray: "_save_numpy"})

    def _save_numpy(self) -> None:
        """
        Save a numpy array to a NPY file.

        See Also
        --------
        :func:`numpy.save`
        """
        np.save(self.path, self.data)


class SaverCSV(Saver):
    """
    Save data in the CSV format.

    Attributes
    ----------
    save_index : bool, default=False
        Flag to determine whether to save the index of a DataFrame)
        (additional attribute compared to the base class).
    """

    ext = FileExt.CSV
    save_methods = MappingProxyType(
        {list: "_save_list", np.ndarray: "_save_numpy", pd.DataFrame: "_save_dataframe"}
    )

    def __init__(self, path: str, data: Any, save_index: bool = False) -> None:
        super().__init__(path, data)
        self.save_index = save_index

    def _save_list(self):
        """Save a list of lists to a CSV file.

        Warning
        -------
        The content of the list is written as strings.

        See Also
        --------
        :func:`csv.writer`
        """
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.data)

    def _save_numpy(self):
        """Save a numpy array to a CSV file.

        Warning
        -------
        The content of the array is written as strings.
        The saved file does not store the original data type,
        so when loading the data back, it is necessary to specify
        the desired data type explicitly.

        See Also
        --------
        :func:`numpy.savetxt`
            Parameter ``fmt``: Format for writing data.
            - ``'%s'``  : strings
            - ``'%d'``  : integers
            - ``'%f'``  : floats
            If not specified, the default format are:
            - ``'%.18e'`` for floating-point numbers
            - string representation for other types
            Thus, differentiation between ``int`` and ``float`` is not automatic.
            Parameter ``delimiter`` : Here, comma (``'``) by default.
        """
        if np.issubdtype(self.data.dtype, np.integer):
            fmt = "%d"
        elif np.issubdtype(self.data.dtype, np.floating):
            fmt = "%.18e"
        else:
            fmt = "%s"
        np.savetxt(self.path, self.data, delimiter=",", fmt=fmt)

    def _save_dataframe(self):
        """Save a pandas DataFrame to a CSV file.

        Notes
        -----
        The index is saved or not based on the `save_index` attribute.
        If True, the DataFrame index is saved as an additional column.
        It is relevant if the index contains meaningful row labels
        (e.g., timestamps, unique identifiers...),
        but not if the index is just a default integer index.
        Dropping the index ensures that the CSV file format is compatible
        with other tools that expect data without an extra index column.

        See Also
        --------
        :meth:`pandas.DataFrame.to_csv`
        """
        self.data.to_csv(self.path, index=self.save_index)
