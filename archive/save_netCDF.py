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
