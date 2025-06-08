"""
`test_core.test_coordinates.test_trials` [module]

See Also
--------
`core.coordinates.trials`: Tested module.

Notes
-----
Those tests focus on the functionalities which are *specific* to
one concrete subclass of coordinates.
"""

import numpy as np
import pytest

from core.coordinates.trial_analysis_label_coord import CoordError, CoordFold


N_SMPL = 10


def test_coord_error_build_labels():
    """
    Test :meth:`CoordError.build_labels`.

    Test Inputs
    -----------
    N_SMPL : 10
    error : True

    Expected Output
    ---------------
    values : np.ndarray
        10 samples of True.
    """
    values = CoordError.build_labels(n_smpl=N_SMPL)
    expected_values = np.full(N_SMPL, False, dtype=np.bool_)
    assert np.array_equal(values, expected_values)


def test_coord_error_count_by_lab():
    """
    Test :meth:`CoordError.count_by_lab`.

    Test Inputs
    -----------
    initial_values : np.ndarray
        5 samples of True, 5 samples of False.

    Expected Output
    ---------------
    count : Dict[bool, int]
        {True: 5, False: 5}
    """
    values = np.array(5 * [True] + 5 * [False], dtype=np.bool_)
    coord = CoordError(values=values)
    count = coord.count_by_lab()
    expected_count = {True: 5, False: 5}
    assert count == expected_count


def test_coord_fold_build_labels():
    """
    Test :meth:`CoordFold.build_labels`.

    Test Inputs
    -----------
    N_SMPL : 10

    Expected Output
    ---------------
    values : np.ndarray
        10 samples of 0.
    """
    values = CoordFold.build_labels(n_smpl=N_SMPL)
    expected_values = np.full(N_SMPL, 0, dtype=np.int64)
    assert np.array_equal(values, expected_values)


# Test values for folds

VALUES = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
COUNT = [2, 2, 2]
TEST_1 = np.array([False, False, True, True, False, False], dtype=np.bool_)


def test_coord_fold_count_by_lab():
    """
    Test :meth:`CoordFold.count_trials`.

    Test Inputs
    -----------
    initial_values : np.ndarray
        2 samples of 0, 2 samples of 1, 2 samples of 2

    Expected Output
    ---------------
    count : List[int]
        [2, 2, 2] : 2 samples of each of the 3 folds.
    """
    coord = CoordFold(values=VALUES)
    count = coord.count_by_lab()
    assert count == COUNT


def test_coord_fold_get_test_train():
    """
    Test :meth:`CoordFold.get_test`.

    Test Inputs
    -----------
    initial_values : np.ndarray
        2 samples of 0, 2 samples of 1
    fold : 2

    Expected Output
    ---------------
    expected_test : np.ndarray
        Boolean mask for the test samples, at the 2nd fold.
    expected_train : np.ndarray
        Boolean mask for the train samples, all except the 2nd fold.
    """
    fold = 1
    coord = CoordFold(values=VALUES)
    mask_test = coord.get_test(fold=fold)
    assert np.array_equal(mask_test, TEST_1)
    expected_train = ~TEST_1
    mask_train = coord.get_train(fold=fold)
    assert np.array_equal(mask_train, expected_train)
