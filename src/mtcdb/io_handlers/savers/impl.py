#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`mtcdb.io_handlers.savers.impl` [module]

Save data from files in specific formats.

Any object which needs to save data can interact with one Saver subclass.

Classes
-------
:class:`SaverPKL`
:class:`SaverNPY`
:class:`SaverCSV`

See Also
--------
:class:`mtcdb.io_handlers.formats.FileExt`: File extensions.
:class:`mtcdb.io_handlers.savers.base.Saver`: Base class for savers.
"""

import pickle
from types import MappingProxyType
from typing import Any

import csv
import numpy as np
import pandas as pd

from mtcdb.io_handlers.formats import FileExt
from mtcdb.io_handlers.savers.base import Saver


class SaverPKL(Saver):
    """
    Save data in the Pickle format.

    Notes
    -----
    Any object can be saved in a Pickle file,
    therefore the class attribute :attr:`save_methods` is not needed.
    Since it is not possible to enumerate all the types that can 
    be saved in a Pickle file, the method :meth:`_save` is overridden
    to dodge the checking step of the base method.
    """
    ext = FileExt.PKL

    def save(self):
        """Save any Python object to a Pickle file.
        
        Warning
        -------
        This method overrides the base method to avoid the checking step.

        See Also
        --------
        :func:`pickle.dump`
        """
        self._check_dir()
        self._check_ext()
        with self.path.open('wb') as file:
            pickle.dump(self.data, file)
        print(f"Saved to {self.path}")


class SaverNPY(Saver):
    """
    Save data in the NPY format.
    """
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
    save_methods = MappingProxyType({
        list: "_save_list",
        np.ndarray: "_save_numpy",
        pd.DataFrame: "_save_dataframe"
    })

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
        with self.path.open('w', newline='', encoding='utf-8') as f:
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
            fmt = '%d'
        elif np.issubdtype(self.data.dtype, np.floating):
            fmt = '%.18e'
        else:
            fmt = '%s'
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



# pylint: disable=W0105
"""
Save data under the netCDF format.

Constraints for NetCDF:
- Data Format : dict 
    Keys   : str, names of variables
    Values : numpy arrays
- Dimension consistency
    The length of each dimension must be consistent across all variables that share that dimension. 
- Data Types : integers, floats, strings...
- Type consistency across dimensions and variables.
- Metadata: Variables and dimensions can have attributes (metadata) attached to them.

Notes
-----
Input data should be a dictionary with the following structure:
- Keys : str, names of variables
- Values : numpy arrays

Implementation
--------------
First, create a new NetCDF file ("dataset") at the specified path.
Second, for each key-value pair in the input data (name-array), 
perform several encoding steps:

1. Create a dimension. Specify two parameters :
    - Name : key (str)
    - Length : len(value) (length of the numpy array)
2. Create a variable. Specify three parameters :
    - Name : key (str)
    - Data type : value.dtype (data type of the numpy array)
    - Dimensions : (key,) (tuple with the dimension name(s))
    Boolean variables are stored as integers.
3. Assign the numpy array to the variable.

Example
-------
.. code-block:: python
    # Example data
    shape = (10, 5, 2) # 3 dimensions
    data_dict = {
        'data': np.random.random(shape),               # 3D array of floats,
        'time': np.linspace(0, 1, shape[0]),           # 1D array of floats for axis 0
        'labels': np.array(['a', 'b', 'c', 'd', 'e']), # 1D array of str for axis 1
        'errors': np.array([True, False]),             # 1D array of bool for axis 2
        'meta1': 'name',
        'meta2': ['info', 'info', 'info']
    }
    # Create a new NetCDF file
    with nc.Dataset('data.nc', 'w', format='NC4') as dataset:
        # Create dimensions
        dataset.createDimension('time', len(time))
        dataset.createDimension('labels', len(labels))
        dataset.createDimension('errors', len(errors))
        # Create variables
        time_var = dataset.createVariable('time', np.float32, ('time',))
        labels_var = dataset.createVariable('labels', str, ('labels',))
        errors_var = dataset.createVariable('errors', np.int32, ('errors',))  # Store bool as int
        data_var = dataset.createVariable('data', np.float32, ('time', 'labels', 'errors'))
        # Assign values
        time_var[:] = data_dict['time']
        labels_var[:] = data_dict['labels']
        errors_var[:] = data_dict['errors'].astype(np.int32) # convert bool to int
        data_var[:, :, :] = data_dict['data']
        # Add metadata
        dataset.meta1 = data_dict['meta1']
        dataset.meta2 = data_dict['meta2']
 """
