#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_processors.test_base` [module]

See Also
--------
:mod:`core.processors.base_processor`: Tested module.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportRedeclaration=false
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access


from typing import Mapping

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from processors.base_processor import Processor


# --- Fixtures -------------------------------------------------------------------------------------


def computation(input_arr: np.ndarray) -> np.ndarray:
    """
    Simple computation performed by a processor: multiply by 2.

    Parameters
    ----------
    input_arr: np.ndarray

    Returns
    -------
    input_arr: np.ndarray
    """
    return input_arr * 2


@pytest.fixture
def input_attrs():
    """
    Fixture - Generate the class attribute :attr:`input_attrs`.

    Returns
    -------
    Tuple[str, ...]:
        Names of the input attributes.
    """
    return ("input1", "input2")


@pytest.fixture
def output_attrs(input_attrs):
    """
    Fixture - Generate the class attribute :attr:`output_attrs`.

    Returns
    -------
    Tuple[str, ...]:
        Names of the output attributes: ("output1", "output2")
    """
    return tuple(f"output{i}" for i in range(1, len(input_attrs) + 1))


@pytest.fixture
def empty_data(input_attrs, output_attrs) -> Mapping[str, np.ndarray]:
    """
    Fixture - Generate the class attribute :attr:`empty_data`.

    Returns
    -------
    MappingProxyType:
        Mapping of dynamic attributes to their corresponding empty state.
    """
    return {attr: np.empty(0) for attr in input_attrs + output_attrs}


@pytest.fixture
def subclass(request):
    """
    Fixture - Define a test class inheriting from :class:`Processor`.

    Returns
    -------
    TestClass:
        Class inheriting from :class:`Processor`. It defines the class attributes `config_attrs`,
        `input_attrs` and `output_attrs` as in the base class. It implements the abstract method
        `_process`.
    """

    class TestClass(Processor):

        config_attrs = ("param",)
        # input_attrs = request.getfixturevalue("input_attrs")
        input_attrs = ("input1", "input2")
        output_attrs = request.getfixturevalue("output_attrs")
        empty_data = request.getfixturevalue("empty_data")

        def __init__(self, param: int = 1):
            super().__init__(param=param)

        def _process(self):
            for attr in self.input_attrs:
                input_arr = getattr(self, attr)
                output_arr = computation(input_arr)
                setattr(self, f"output{attr[-1]}", output_arr)
                setattr(self, attr, self.empty_data[attr])

    return TestClass


@pytest.fixture
def input_data(input_attrs):
    """
    Fixture - Generate input data for the processing test.

    Returns
    -------
    input_data: Dict[str, np.ndarray]
        Input data to pass to the processor.
    """
    return {f"input{i}": np.arange(i, i + 3) for i in range(1, len(input_attrs) + 1)}


@pytest.fixture
def output_data(input_data):
    """
    Fixture - Generate expected output data corresponding to the input data (multiply by 2)

    Returns
    -------
    expected_output: Dict[str, np.ndarray]
        Expected output from the processor after applying the processing logic.
    """
    return {
        f"output{i}": computation(input_data[f"input{i}"]) for i in range(1, len(input_data) + 1)
    }


# --- Tests ----------------------------------------------------------------------------------------


def test_init_config(subclass):
    """
    Test the initialization of the configuration parameters by the base class constructor.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Processor`.

    Expected Output
    ---------------
    Input and output properties should be accessible via the dynamic attribute names.
    """
    processor = subclass(param=42)
    assert hasattr(processor, "param"), f"Missing config attribute: 'param'"


def test_init_empty_data(input_attrs, output_attrs, subclass):
    """
    Ensure that the data attributes are initialize with default empty state.

    Expected Output
    ---------------
    Input and output attributes should be set to their empty state (e.g., empty arrays).
    """
    expected_empty_array = np.empty(0)  # empty array to compare with reset attributes
    processor = subclass()
    # Check attribute existence
    for attr in input_attrs + output_attrs:
        assert hasattr(processor, attr), f"Missing attribute: {attr}"
    # Check initial empty state
    for attr in input_attrs + output_attrs:
        assert_array_equal(getattr(processor, attr), expected_empty_array)
    # Check that the flag has been reset to False
    assert processor._has_input is False
    assert processor._has_output is False


def test_validate(subclass, input_data):
    """
    Test the method :meth:`_validate` for basic validation logic.

    Expected Output
    ---------------
    ValueError raised for missing input attributes or unexpected inputs.
    """
    processor = subclass()
    # Missing input: remove one required element from input_data
    key1 = list(input_data.keys())[0]
    input_data_missing = {k: v for k, v in input_data.items() if k != key1}
    with pytest.raises(ValueError):
        processor._validate(**input_data_missing)
    # Unexpected input: add an extra element to input_data
    input_data_extra = {**input_data, "extra": np.array([1, 2, 3])}
    with pytest.raises(ValueError):
        processor._validate(**input_data_extra)


def test_reset(subclass, input_data):
    """
    Test the method :meth:`_reset` for clearing the input and output data.

    Expected Output
    ---------------
    Input and output attributes should be reset to their empty state (e.g., empty arrays).
    """
    input_array = np.array([1, 2, 3])
    input_attr = "input1"
    expected_empty_array = np.empty(0)  # empty array to compare with reset attributes
    # Set input1 and flag manually
    processor = subclass()
    setattr(processor, input_attr, input_array)
    processor._has_input = True
    assert_array_equal(getattr(processor, input_attr), input_array)
    # Check input attributes and flag after reset
    processor._reset()
    assert_array_equal(getattr(processor, input_attr), expected_empty_array)
    assert processor._has_input is False


def test_set_inputs(subclass, input_data):
    """
    Test the method :meth:`_set_inputs` for setting input data.

    Expected Output
    ---------------
    Input attributes should be set to the provided data.
    """
    processor = subclass()
    processor._set_inputs(**input_data)
    for key, value in input_data.items():
        assert_array_equal(getattr(processor, key), value)
    assert processor._has_input is True


def test_check_outputs(subclass, input_data, output_data):
    """
    Test the method :meth:`_check_outputs` for checking the output data.

    Expected Output
    ---------------
    If outputs are missing, raise a ValueError.
    If outputs are present, the flag `_has_output` should be set to True.
    """
    processor = subclass()
    # Check missing outputs (no process is called, empty data is set)
    with pytest.raises(ValueError):
        processor._check_outputs()
    # Set outputs manually
    for key, value in output_data.items():
        setattr(processor, key, value)
    # Check outputs and flag
    processor._check_outputs()
    assert processor._has_output is True


def test_process(subclass, input_data, output_data):
    """
    Test the :meth:`process` method inherited from the base class.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Processor`.
    input_data: Dict[str, np.ndarray]
        Input data passed to the processor.
    output_data: Dict[str, np.ndarray]
        Expected output from the processor after applying the processing logic.

    Expected Output
    ---------------
    Store the processed output among the attributes.
    """
    processor = subclass()
    processor.process(**input_data)
    expected_output = output_data
    # Check correct outputs
    for key in expected_output:
        np.testing.assert_array_equal(getattr(processor, key), expected_output[key])
    # Validate the internal data flag
    assert processor._has_output is True


def test_seed_property(subclass, input_data, output_data):
    """
    Test the behavior of setting and getting the seed property.

    Expected Output
    ---------------
    - Getting the seed returns the correct value.
    - Setting the seed clears the output data.
    """
    new_seed = 42
    processor = subclass()
    processor.process(**input_data)
    # Ensure outputs are initially populated
    assert processor._has_output is True
    for output_attr, expected_value in output_data.items():
        np.testing.assert_array_equal(getattr(processor, output_attr), expected_value)
    # Set seed manually
    processor.seed = new_seed
    assert processor.seed == new_seed
    # Ensure output data is cleared and flag is reset
    assert processor._has_output is False
    for output_attr in processor.output_attrs:
        np.testing.assert_array_equal(
            getattr(processor, output_attr), processor.empty_data[output_attr]
        )


def test_set_random_state(subclass, input_data):
    """
    Test the method :meth:`set_random_state` for setting the random state.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Processor`.
    input_data: Dict[str, np.ndarray]
        Input data passed to the processor.
    seed: int
        Random seed value.

    Expected Output
    ---------------
    Seed should be stored in the processor.
    """
    processor = subclass()
    seed = 42
    processor.process(**input_data, seed=seed)  # run process with a specific seed
    assert processor.seed == seed
    assert processor._has_output is True
