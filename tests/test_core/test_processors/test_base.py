#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_processors.test_base` [module]

See Also
--------
:mod:`core.processors.base`: Tested module.
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
import pytest

from core.processors.base import Processor


# --- Fixtures -------------------------------------------------------------------------------------


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
def output_attrs():
    """
    Fixture - Generate the class attribute :attr:`output_attrs`.

    Returns
    -------
    Tuple[str, ...]:
        Names of the output attributes.
    """
    return ("output1", "output2")


@pytest.fixture
def optional_attrs():
    """
    Fixture - Generate the class attribute :attr:`optional_attrs`.

    Returns
    -------
    Tuple[str, ...]:
        Names of the optional attributes.
    """
    return ("optional_input",)


@pytest.fixture
def proc_data_type(input_attrs, output_attrs) -> Mapping[str, type]:
    """
    Fixture - Generate the class attribute :attr:`proc_data_type`.

    Returns
    -------
    MappingProxyType:
        Mapping of dynamic attributes to their corresponding types.
    """
    return {attr: np.ndarray for attr in (*input_attrs, *output_attrs)}


@pytest.fixture
def proc_data_empty(input_attrs, output_attrs) -> Mapping[str, np.ndarray]:
    """
    Fixture - Generate the class attribute :attr:`proc_data_empty`.

    Returns
    -------
    MappingProxyType:
        Mapping of dynamic attributes to their corresponding empty state.
    """
    return {attr: np.empty(0) for attr in (*input_attrs, *output_attrs)}


@pytest.fixture
def subclass(request):
    """
    Fixture - Define a test class inheriting from :class:`Processor`.

    Returns
    -------
    TestClass:
        Class inheriting from :class:`Processor`. It defines the class attributes `input_attrs`,
        `output_attrs`, and `proc_data_type` required by the metaclass :class:`ProcessorMeta`. It
        implements the abstract method `_process`.
    """

    class TestClass(Processor):

        input_attrs = request.getfixturevalue("input_attrs")
        output_attrs = request.getfixturevalue("output_attrs")
        proc_data_empty = request.getfixturevalue("proc_data_empty")

        def _process(self):
            result1 = self.input1 * 2
            result2 = self.input2 + 3
            return {"output1": result1, "output2": result2}

    return TestClass


@pytest.fixture
def input_data():
    """
    Fixture - Generate input data for the processing test.

    Returns
    -------
    input_data: Dict[str, np.ndarray]
        Input data passed to the processor.
    expected_output: Dict[str, np.ndarray]
        Expected output from the processor after applying the processing logic.
    """
    return {
        "input1": np.array([1, 2, 3]),
        "input2": np.array([4, 5, 6]),
    }


@pytest.fixture
def output_data():
    """
    Fixture - Generate expected output data corresponding to the input data.

    Returns
    -------
    expected_output: Dict[str, np.ndarray]
        Expected output from the processor after applying the processing logic.
    """
    return {
        "output1": np.array([2, 4, 6]),  # input1 * 2
        "output2": np.array([7, 8, 9]),  # input2 + 3
    }


# --- Tests ----------------------------------------------------------------------------------------


def test_class_creation(input_attrs, output_attrs, subclass):
    """
    Test subclass creation involving :class:`ProcessorMeta` and :class:`Processor`.

    Expected Output
    ---------------
    Class attributes :attr:`input_attrs` and :attr:`output_attrs` should be set.
    """
    # Check the presence of the class attributes
    assert hasattr(subclass, "input_attrs")
    assert hasattr(subclass, "output_attrs")
    assert hasattr(subclass, "intermediate_attrs")  # not explicitly defined in concrete subclass
    assert hasattr(subclass, "proc_data_type")
    # Check the content of class input/output attributes
    assert subclass.input_attrs == input_attrs
    assert subclass.output_attrs == output_attrs


def test_properties(subclass):
    """
    Test the creation of properties for input and output attributes by the metaclass.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Processor`.

    Expected Output
    ---------------
    Input and output properties should be accessible via the dynamic attribute names.
    """
    processor = subclass()
    for input_attr in subclass.input_attrs:
        assert hasattr(processor, input_attr), f"Missing property (input attr): {input_attr}"
    for output_attr in subclass.output_attrs:
        assert hasattr(processor, output_attr), f"Missing property (output attr): {output_attr}"


def test_check_consistency():
    """
    Test that :meth:`ProcessorMeta.check_consistency` detects inconsistencies between dynamic
    attributes and `proc_data_type`.

    Define an invalid test class inheriting from :class:`Processor`, where the class attributes
    `input_attrs` and `output_attrs` do not match the `proc_data_type` mapping.

    Expected Output
    ---------------
    ValueError raised for a mismatch between the dynamic attributes in `input_attrs`,
    `output_attrs`, and the keys in `proc_data_type`.
    """
    with pytest.raises(ValueError):

        class InconsistentProcessor(Processor):
            input_attrs = ("input1", "input2")
            output_attrs = ("output1",)
            proc_data_type = {  # missing 'input2', extra 'output2'
                "input1": np.ndarray,
                "output1": np.ndarray,
                "output2": np.ndarray,
            }


def test_init_empty_data(subclass):
    """
    Ensure that the data attributes are initialize with default empty state.

    Expected Output
    ---------------
    Dynamic attributes should be set to their empty state (e.g., empty arrays).
    """
    expected_empty_array = np.empty(0)  # empty array to compare with reset attributes
    processor = subclass()
    # Check initial empty state
    for input_attr in subclass.input_attrs:
        np.testing.assert_array_equal(getattr(processor, f"_{input_attr}"), expected_empty_array)
    # Check that the flag has been reset to False
    assert processor._has_output is False


def test_check_name_type(subclass, proc_data_empty):
    """
    Test the :meth:`_check_name_type` method for checking inputs data names and types.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Processor`.
    proc_data_empty: Mapping[str, np.ndarray]
        Mapping of dynamic attributes to their corresponding empty state.

    Expected Output
    ---------------
    ValueError raised for invalid data types in dynamic attributes.
    """
    processor = subclass()
    # Test valid data types
    for attr, value in proc_data_empty.items():
        processor._check_name_type(attr, value)
    # Test invalid data type
    with pytest.raises(TypeError):
        for attr, _ in proc_data_empty.items():
            processor._check_name_type(attr, "invalid_type")
    # Test invalid setting of dynamic attribute (wrong name)
    with pytest.raises(ValueError):
        for attr, value in proc_data_empty.items():
            processor._check_name_type("invalid_name", value)


def test_set_data(subclass):
    """
    Test the :meth:`set_data` method for assigning dynamic data to internal attributes.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Processor`.

    Expected Output
    ---------------
    Dynamic attributes should be set, and invalid assignments should raise errors.
    """
    input_arr = np.array([1, 2, 3])
    processor = subclass()
    processor._set_data("input1", input_arr)
    np.testing.assert_array_equal(processor._input1, input_arr)


def test_check_missing(subclass, input_data):
    """
    Test the method :meth:`_check_missing` for validation logic.

    Expected Output
    ---------------
    ValueError raised for missing input attributes.
    """
    processor = subclass()
    # Test missing input: remove one required element from input_data
    key1 = list(input_data.keys())[0]
    input_data_invalid = {k: v for k, v in input_data.items() if k != key1}
    with pytest.raises(ValueError):
        processor._check_missing(input_data_invalid, processor.input_attrs)


def test_process(subclass, input_data, output_data, optional_attrs):
    """
    Test the :meth:`process` method in the concrete subclass.

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
    The processed output should be stored in the internal attributes and returned.
    """
    processor = subclass()
    output = processor.process(**input_data)
    expected_output = output_data
    # Check correct outputs
    for key in expected_output:
        np.testing.assert_array_equal(output[key], expected_output[key])
        np.testing.assert_array_equal(getattr(processor, f"_{key}"), expected_output[key])
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
        np.testing.assert_array_equal(getattr(processor, f"_{output_attr}"), expected_value)
    # Set seed manually
    processor.seed = new_seed
    assert processor.seed == new_seed
    # Ensure output data is cleared and flag is reset
    assert processor._has_output is False
    for output_attr in processor.output_attrs:
        np.testing.assert_array_equal(
            getattr(processor, f"_{output_attr}"), processor.proc_data_empty[output_attr]
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
