#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_coordinates.test_exp_condition` [module]

See Also
--------
:mod:`mtcdb.coordinates.exp_condition`: Tested module.

Notes
-----
Those tests focus on the functionalities which are *specific* to 
one concrete subclass of coordinates.

The tests are performed on the :class:`CoordTask`subclass, 
but apply for all subclasses of :class:`CoordExpCond`.
"""

import numpy as np
import pytest # pylint: disable=unused-import

from mtcdb.coordinates.exp_condition import CoordTask
from mtcdb.core_objects.exp_condition import Task


N_SMPL = 10


def test_coord_task_build_labels():
    """
    Test :meth:`CoordTask.build_labels`.

    Test Inputs
    -----------
    N_SMPL : 10
    cnd : Task('PTD')

    Expected Output
    ---------------
    values : np.ndarray
        10 samples of 'PTD'.
    """
    values = CoordTask.build_labels(n_smpl=N_SMPL, cnd=Task('PTD'))
    expected_values = np.full(N_SMPL, 'PTD')
    assert np.array_equal(values, expected_values)


def test_coord_task_replace_label():
    """
    Test :meth:`CoordTask.replace_label`.

    Test Inputs
    -----------
    initial_values : 10 samples of 'CCH'.
    old : Task('CCH')
    new : Task('PTD')

    Expected Output
    ---------------
    updated_values : 10 samples of 'PTD'.
    """
    old_values = np.full(N_SMPL, 'CCH')
    new_values = np.full(N_SMPL, 'PTD')
    old_coord = CoordTask(values=old_values)
    new_coord = old_coord.replace_label(old=Task('CCH'), new=Task('PTD'))
    assert np.array_equal(new_coord.values, new_values)


def test_task_count_by_lab():
    """
    Test :meth:`CoordTask.count_by_lab`.

    Test Inputs
    -----------
    initial_values : np.ndarray
        5 samples of 'PTD', 5 samples of 'CLK'.

    Expected Output
    ---------------
    count : Dict[Task, int]
        {Task('PTD'): 5, Task('CLK'): 5}
    """
    values = np.array(5*['PTD'] + 5*['CLK'])
    coord = CoordTask(values=values)
    count = coord.count_by_lab()
    expected_count = {Task('PTD'): 5, Task('CLK'): 5, Task('CCH'): 0}
    assert count == expected_count
