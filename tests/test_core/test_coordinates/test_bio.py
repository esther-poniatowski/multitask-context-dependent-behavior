#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_coordinates.test_bio` [module]

See Also
--------
:mod:`core.coordinates.bio`: Tested module.
:class:`core.attributes.bio_data.Depth`
:class:`core.attributes.composites.Unit`
"""

import numpy as np
import pytest

from core.coordinates.bio_info_coord import CoordUnit, CoordDepth
from core.attributes.bio_info import CorticalDepth
from core.attributes.composites import Unit


# Test values for CoordUnit
UNITS_IDS = ["avo052a-d1", "daf035b-d2"]
DEPTHS = ["a", "b"]  # letter after the first number in the unit ID
POP = [Unit(unit_id) for unit_id in UNITS_IDS]


def test_coord_unit_build_labels_valid():
    """
    Test :meth:`CoordUnit.build_labels` with a valid container of units.

    Test Inputs
    -----------
    units : List of two units. Here, units IDs are arbitrary.

    Expected Output
    ---------------
    values : Array containing the string labels of the units.
    """
    values = CoordUnit.build_labels(units=POP)
    expected_values = np.array(UNITS_IDS, dtype=np.str_)
    assert np.array_equal(values, expected_values)


def test_coord_unit_build_labels_invalid():
    """
    Test :meth:`CoordUnit.build_labels` with an invalid container of units.

    Test Inputs
    -----------
    invalid_input : List of strings, which are not of type Unit.

    Expected Output
    ---------------
    TypeError
    """
    with pytest.raises(TypeError):
        CoordUnit.build_labels(units=UNITS_IDS)


def test_coord_depth_build_labels():
    """
    Test :meth:`CoordDepth.build_labels` with a valid container of units.

    Test Inputs
    -----------
    units : List of two units.

    Expected Output
    ---------------
    values : Array containing the depths of the units.
    """
    values = CoordDepth.build_labels(units=POP)
    expected_values = np.array(DEPTHS, dtype=np.str_)
    assert np.array_equal(values, expected_values)


# Test values for CoordDepth
VALUES = np.array(["a", "a", "b", "b"], dtype=np.str_)


def test_coord_depth_get_layer():
    """
    Test :meth:`CoordDepth.get_layer`.

    Test Inputs
    -----------
    values : array('a' 'a', 'b', 'b')
    depth : CorticalDepth('b')

    Expected Output
    ---------------
    mask : np.ndarray
        Boolean mask for the samples in the layer 'b'.
    """
    coord = CoordDepth(values=VALUES)
    mask = coord.get_layer(depth=CorticalDepth("b"))
    expected_mask = np.array(
        [
            False,
            False,
            True,
            True,
        ],
        dtype=np.bool_,
    )
    assert np.array_equal(mask, expected_mask)


def test_coord_depth_count_by_lab():
    """
    Test :meth:`CoordDepth.count_by_lab`.

    Test Inputs
    -----------
    values : array('a' 'a', 'b', 'b')

    Expected Output
    ---------------
    count : Dict[CorticalDepth, int]
        {CorticalDepth('a'): 2, CorticalDepth('b'): 2}
    """
    coord = CoordDepth(values=VALUES)
    count = coord.count_by_lab()
    expected_count = {
        CorticalDepth("a"): 2,
        CorticalDepth("b"): 2,
        CorticalDepth("c"): 0,
        CorticalDepth("d"): 0,
        CorticalDepth("e"): 0,
        CorticalDepth("f"): 0,
    }
    assert count == expected_count


if __name__ == "__main__":
    pytest.main()
