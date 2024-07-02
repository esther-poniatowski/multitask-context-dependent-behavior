#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_mtcdb.test_utils.test_objects` [module]

Tests for the module :mod:`mtcdb.core_objects.coordinates`.
"""

import numpy as np
import pytest

from mtcdb.core_objects.coordinates import CoordTime
from mtcdb.core_objects.coordinates import CoordTask, CoordContext, CoordStim
from mtcdb.core_objects.coordinates import CoordRecNum, CoordBlock, CoordSlot, CoordFold
from mtcdb.core_objects.coordinates import CoordUnit, CoordDepth
from mtcdb.core_objects.expt_obj import Task, Context, Stimulus, Recording, Block, Slot, Unit, CorticalDepth


def test_coord_time_init_with_values():
    values = np.array([0, 0.1, 0.2], dtype=np.float64)
    coord_time = CoordTime(values=values)
    assert coord_time.tbin == 0.1
    assert coord_time.n_t == 3


def test_coord_time_init_with_nt():
    coord_time = CoordTime(n_t=3, tbin=0.1)
    expected_values = np.array([0, 0.1, 0.2], dtype=np.float64)
    assert np.allclose(coord_time.values, expected_values)


def test_coord_task_init_with_values():
    values = np.array(['PTD', 'CLK'])
    coord_task = CoordTask(values=values)
    assert np.array_equal(coord_task.values, values)


def test_coord_context_init_with_values():
    values = np.array(['a', 'p'])
    coord_context = CoordContext(values=values)
    assert np.array_equal(coord_context.values, values)


def test_coord_stim_init_with_values():
    values = np.array(['R', 'T', 'N'])
    coord_stim = CoordStim(values=values)
    assert np.array_equal(coord_stim.values, values)

def test_replace_label():
    class MockCondition:
        def __init__(self, value):
            self.value = value
    initial_values = np.array(['p-pre', 'a', 'p'], dtype=np.str_)
    coord = CoordContext(values=initial_values)
    old = MockCondition('p-pre')
    new = MockCondition('p')
    updated_values = coord.replace_label(old, new)
    expected_values = np.array(['p', 'a', 'p'], dtype=np.str_)
    assert np.array_equal(updated_values, expected_values)


def test_coord_rec_num_count_trials():
    class MockRecording:
        def __init__(self, pos):
            self.pos = pos
    initial_values = np.array([1, 2, 1, 3, 2, 1], dtype=np.int64)
    coord_rec_num = CoordRecNum(values=initial_values)
    counts = coord_rec_num.count_trials()
    expected_counts = {1: 3, 2: 2, 3: 1}
    assert counts == expected_counts


def test_coord_fold_init():
    coord_fold = CoordFold(n_tr=10)
    expected_values = np.zeros(10, dtype=np.int64)
    assert np.array_equal(coord_fold.values, expected_values)


def test_coord_fold_count_trials():
    values = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    coord_fold = CoordFold(values=values)
    assert coord_fold.count_trials() == [2, 2, 3, 3]


def test_coord_unit_init_with_values():
    values = np.array(['unit1', 'unit2'])
    coord_unit = CoordUnit(values=values)
    assert np.array_equal(coord_unit.values, values)


def test_coord_depth_init_with_values():
    values = np.array(['a', 'b', 'c'])
    coord_depth = CoordDepth(values=values)
    assert np.array_equal(coord_depth.values, values)


def test_coord_unit_init_with_units():
    class MockUnit:
        def __init__(self, id):
            self.id = id
            
    units = [MockUnit('unit1'), MockUnit('unit2')]
    coord_unit = CoordUnit(units=units)
    expected_values = np.array(['unit1', 'unit2'], dtype=np.str_)
    print(f"coord_unit.values: {coord_unit.values}, expected_values: {expected_values}")
    assert np.array_equal(coord_unit.values, expected_values)


def test_coord_depth_init_with_units():
    class MockDepth:
        def __init__(self, value):
            self.value = value
            
    class MockUnit:
        def __init__(self, depth):
            self.depth = depth
            
    units = [MockUnit(MockDepth('depth1')), MockUnit(MockDepth('depth2'))]
    coord_depth = CoordDepth(units=units)
    expected_values = np.array(['depth1', 'depth2'], dtype=np.str_)
    print(f"coord_depth.values: {coord_depth.values}, expected_values: {expected_values}")
    assert np.array_equal(coord_depth.values, expected_values)


def test_count_layers():
    initial_values = np.array(['a', 'b', 'b', 'c', 'c', 'c'], dtype=np.str_)
    coord_depth = CoordDepth(values=initial_values)
    counts = coord_depth.count_layers()
    expected_counts = {
        CorticalDepth('a'): 1,
        CorticalDepth('b'): 2,
        CorticalDepth('c'): 3,
        CorticalDepth('d'): 0,
        CorticalDepth('e'): 0,
        CorticalDepth('f'): 0
    }
    assert counts == expected_counts


if __name__ == '__main__':
    pytest.main()