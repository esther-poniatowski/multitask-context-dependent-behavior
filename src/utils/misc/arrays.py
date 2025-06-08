"""
:mod:`utils.misc.arrays` [module]

Utilities for arrays.

Functions
---------
:func:`unique`

"""
import numpy as np
import numpy.typing as npt


def create_empty_array(n_dims: int, dtype: npt.DTypeLike) -> np.ndarray:
    """
    Create an empty array with a given number of dimensions and data type.

    Each dimension has size zero (i.e. length 0).

    Parameters
    ----------
    n_dims: int
        Number of dimensions.
    dtype: numpy.dtype
        Data type.

    Returns
    -------
    numpy.ndarray
        Empty array with the specified number of dimensions and data type.
    """
    return np.empty((0,) * n_dims, dtype=dtype)
