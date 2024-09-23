#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`test_core.test_processors.test_base_processor` [module]

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
# pylint: disable=unused-argument
# pylint: disable=expression-not-assigned
# mypy: disable-error-code="annotation-unchecked"

from dataclasses import dataclass, fields

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from core.processors.base_processor import Processor, ProcessorInput, ProcessorOutput


# --- Fixtures for a simple processor --------------------------------------------------------------


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
def subclass_inputs():
    """
    Fixture - Define a data class for the class attribute :attr:`input_dataclass`.

    Returns
    -------
    TestClassInputs:
        Data class for the input attributes, inheriting from :class:`ProcessorInput`.
        Its attributes correspond to the valid input names.
        It implements the method `validate` just to check that it is possible to pass configuration
        parameters from the associated processor subclass.
    """

    @dataclass
    class TestClassInputs(ProcessorInput):
        input1: np.ndarray
        input2: np.ndarray

        def validate(self, param=1, **config_params):
            l = len(self.input1)
            if l < param:
                raise ValueError(f"Invalid length : {l} < {param}")

    return TestClassInputs


@pytest.fixture
def subclass_outputs():
    """
    Fixture - Define a data class for the class attribute :attr:`output_dataclass`.

    Returns
    -------
    TestClassOutputs:
        Data class for the output attributes, inheriting from :class:`ProcessorOutput`.
        Its attributes correspond to the valid output names.
    """

    @dataclass
    class TestClassOutputs(ProcessorOutput):
        output1: np.ndarray
        output2: np.ndarray

    return TestClassOutputs


@pytest.fixture
def subclass(request, subclass_inputs, subclass_outputs):
    """
    Fixture - Define a processor class implementing a simple computation.

    Returns
    -------
    TestClass:
        Class inheriting from :class:`Processor`.

    Notes
    -----
    To test the main functionalities of the `Processor` class:``

    - Define the class attributes `config_params` with a single parameter `param`. Call the parent
      constructor and pass it as a named argument. Give a default value to facilitate tests.
    - Store a reference to the input and output data classes in the class attributes
      `input_dataclass` and `output_dataclass`.
    - Set the flag `is_random` to `True` to allow random state initialization.
    - Implement the abstract method `_process`, which applies the basic computation defined in the
      fixture `computation`.
    - Do not override the optional methods  `_pre_process` and `_post_process` in order to test
      their base implementation.

    See Also
    --------
    :meth:`request.getfixturevalue`
        Access to fixtures in the current module. Here, it is necessary to store the results of the
        fixtures instead of calling them directly.
    """

    class TestClass(Processor):
        config_params = ("param",)
        input_dataclass = request.getfixturevalue("subclass_inputs")
        output_dataclass = request.getfixturevalue("subclass_outputs")
        is_random = True

        def __init__(self, param: int = 1):
            super().__init__(param=param)

        def _process(self, input1=None, input2=None, **kwargs):
            output1 = computation(input1)
            output2 = computation(input2)
            return output1, output2

    return TestClass


@pytest.fixture
def input_data(subclass_inputs):
    """
    Fixture - Generate valid input data matching the input data class.

    Returns
    -------
    input_data: Dict[str, np.ndarray]
        Input data to pass to the processor.
        Keys: Names of the attributes in the data class `subclass_inputs`.
        Values: Arrays of shape (3,) with increasing values.
    """
    keys = [field.name for field in fields(subclass_inputs)]
    values = [np.arange(i, i + 3) for i in range(len(keys))]
    return dict(zip(keys, values))


@pytest.fixture
def input_missing(input_data):
    """
    Fixture - Generate input data with a missing key.

    Returns
    -------
    Dict[str, np.ndarray]:
        Input data with one key removed.
    """
    key1 = list(input_data.keys())[0]
    return {k: v for k, v in input_data.items() if k != key1}


@pytest.fixture
def input_extra(input_data):
    """
    Fixture - Generate input data with an extra key.

    Returns
    -------
    Dict[str, np.ndarray]:
        Input data with one additional key.
    """
    return {**input_data, "extra": np.array([1, 2, 3])}


@pytest.fixture
def output_names(subclass_outputs):
    """
    Fixture - Generate valid names for the outputs, matching the output data class.

    Returns
    -------
    Tuple[str, ...]:
        Names of the output attributes.
    """
    return tuple([field.name for field in fields(subclass_outputs)])


@pytest.fixture
def output_data(input_data):
    """
    Fixture - Generate expected output data corresponding to the input data (multiply by 2)

    Returns
    -------
    expected_output: Tuple[np.ndarray]
        Expected output from the processor after applying the processing logic.
    """
    return tuple([computation(value) for key, value in input_data.items()])


@pytest.fixture
def output_missing(output_data):
    """
    Fixture - Generate output data with a missing element.

    Returns
    -------
    Tuple[np.ndarray]:
        Output data with one element removed.
    """
    return output_data[1:]


@pytest.fixture
def output_extra(output_data):
    """
    Fixture - Generate output data with an extra element.

    Returns
    -------
    Tuple[np.ndarray]:
        Output data with one additional element (last element, thereby occurring twice).
    """
    return output_data + (output_data[-1],)


# --- Fixtures for a processor to test edge cases --------------------------------------------------


@pytest.fixture
def subclass_inputs_with_default():
    """
    Fixture - Define a data class for the class attribute :attr:`input_dataclass` with one default
    value.

    Returns
    -------
    TestClassInputsWithDefault
    """

    @dataclass
    class TestClassInputsWithDefault(ProcessorInput):
        input1: str = "default"

    return TestClassInputsWithDefault


@pytest.fixture
def subclass_outputs_single_value():
    """
    Fixture - Define a data class for the class attribute :attr:`output_dataclass` which computes a
    single output.

    Returns
    -------
    TestClassOutputsSingleValue
    """

    @dataclass
    class TestClassOutputsSingleValue(ProcessorOutput):
        output1: str

    return TestClassOutputsSingleValue


@pytest.fixture
def single_output_value():
    """
    Fixture - Generate a single output value for the edge case.

    Returns
    -------
    str
    """
    return "unique_output"


@pytest.fixture
def subclass_edge_cases(
    request, subclass_inputs_with_default, subclass_outputs_single_value, single_output_value
):
    """
    Fixture - Define a processor class to test edge cases.

    Returns
    -------
    TestClassEdgeCases
    """

    class TestClassEdgeCases(Processor):
        input_dataclass = request.getfixturevalue("subclass_inputs_with_default")
        output_dataclass = request.getfixturevalue("subclass_outputs_single_value")

        def _process(self, input1=None, input2=None, **kwargs):
            output1 = request.getfixturevalue("single_output_value")
            return output1

    return TestClassEdgeCases


# --- Tests for with a simple processor ------------------------------------------------------------


def test_processor_input(subclass_inputs, input_data):
    """
    Test the input data class for the processor when passing inputs by unpacking a dictionary.

    Test Inputs
    -----------
    subclass_inputs:
        Data class for the input attributes, inheriting from :class:`ProcessorInput`.
    input_data: Dict[str, np.ndarray]
        Input data passed to the data class.
    param: int
        Configuration parameter to pass to the `validate` method, corresponding to the minimal length
        of the input array.

    Expected Output
    ---------------
    The data class instance should have attributes corresponding to the inputs.
    Its method `validate` should pass when `param = 1` but raise an error when `param = 10`, since
    the length of the input array is 3.
    """
    inputs_obj = subclass_inputs(**input_data)
    for name in input_data:
        assert hasattr(inputs_obj, name), f"Missing attribute: '{name}'"
        assert_array_equal(getattr(inputs_obj, name), input_data[name]), "Invalid value"
    inputs_obj.validate(param=1)
    with pytest.raises(ValueError):
        inputs_obj.validate(param=10)


def test_processor_output(subclass_outputs, output_data, output_names):
    """
    Test the output data class for the processor when passing outputs by unpacking a tuple.

    Test Inputs
    -----------
    subclass_outputs:
        Data class for the output attributes, inheriting from :class:`ProcessorOutput`.
    output_data: Tuple[np.ndarray]
        Output data passed to the data class.
    output_names: Tuple[str]
        Names of the expected output attributes, in the order in which they are passed.

    Expected Output
    ---------------
    The data class instance should have attributes corresponding to the outputs.
    """
    outputs_obj = subclass_outputs(*output_data)
    for name, value in zip(output_names, output_data):
        assert hasattr(outputs_obj, name), f"Missing attribute: '{name}'"
        assert_array_equal(getattr(outputs_obj, name), value), "Invalid value"


def test_init_config(subclass):
    """
    Test the initialization of the configuration parameters by the base class constructor when
    passing named arguments.

    Test Inputs
    -----------
    subclass:
        Class inheriting from :class:`Processor`.

    Expected Output
    ---------------
    Input and output properties should be accessible via the dynamic attribute names.
    """
    param = 42
    processor = subclass(param=param)
    assert hasattr(processor, "param"), "Missing config attribute: 'param'"
    assert processor.param == param
    assert processor.seed is None


def test_pre_process(subclass, input_data, input_missing, input_extra):
    """
    Test the base implementation of method :meth:`_pre_process` for validation logic.

    Test Inputs
    -----------
    input_data: Dict[str, np.ndarray]
        Complete and valid input data passed to the method by unpacking the dictionary.
    input_missing: Dict[str, np.ndarray]
        Input data with one missing element (one key removed).
    input_extra: Dict[str, np.ndarray]
        Input data with one extra element (one key added).

    Expected Output
    ---------------
    When inputs are complete, the method should return a dictionary with the input data, validated
    through the data class and converted back to a dictionary.
    ValueError raised for missing input attributes or unexpected inputs.
    """
    processor = subclass()
    # Pass valid input data by unpacking
    validated = processor._pre_process(**input_data)
    assert isinstance(validated, dict)
    # Missing input
    with pytest.raises(ValueError):
        processor._pre_process(**input_missing)
    # Unexpected input
    with pytest.raises(ValueError):
        processor._pre_process(**input_extra)


def test_process(subclass, input_data, output_data):
    """
    Test the :meth:`_process` method implemented by the subclass.

    Test Inputs
    -----------
    input_data: Dict[str, np.ndarray]
        Valid input data passed to the method by unpacking the dictionary.

    Expected Output
    ---------------
    The output should be a tuple of the processed data, as defined in the subclass.
    The order of the elements should match the order of the output attributes.
    """
    processor = subclass()
    out = processor._process(**input_data)
    assert isinstance(out, tuple), f"Output not a tuple, actual type: {type(out)}"
    for actual, expected in zip(out, output_data):
        assert_array_equal(actual, expected), "Invalid output"


def test_post_process(subclass, output_data, output_missing, output_extra):
    """
    Test the base implementation of the :meth:`_post_process` method for validation logic.

    Test Inputs
    -----------
    output_data: Tuple[np.ndarray]
        Expected output data passed to the method by unpacking the tuple.
    output_missing: Tuple[np.ndarray]
        Output data with one missing element (one element removed).
    output_extra: Tuple[np.ndarray]
        Output data with one extra element (one element added).

    Expected Output
    ---------------
    When outputs are valid, the method should return a tuple with the output data, validated
    through the data class and converted back to a tuple.
    ValueError raised for missing, unexpected or invalid outputs.
    """
    processor = subclass()
    # Pass valid input data by unpacking
    validated = processor._post_process(*output_data)
    assert isinstance(validated, tuple)
    # Missing output
    with pytest.raises(ValueError):
        processor._post_process(*output_missing)
    # Unexpected output
    with pytest.raises(ValueError):
        processor._post_process(*output_extra)


def test_process_base(subclass, input_data, output_data):
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
    out = processor.process(**input_data)
    for actual, expected in zip(out, output_data):
        assert_array_equal(actual, expected), "Invalid output"


def test_set_random_state(subclass):
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
    processor.set_random_state(seed)  # run process with a specific seed
    assert processor.seed == seed


# --- Tests edge cases -----------------------------------------------------------------------------


def test_process_with_single_input(subclass_edge_cases, single_output_value):
    """
    Test the processor for edge cases.

    - No inputs is passed to the `process` method to test whether the default value is used after
      the `_pre_process` method is called in the pipeline.
    - The outputs should be validated by the `_post_process` method although there is a single
      value, since it should have been converted into a tuple before calling the `_post_process`
      method. However, the return value should not be a tuple anymore since the single value should
      have been extracted from the tuple after the `_post_process` method.
    """
    # Call the processor without any input
    processor = subclass_edge_cases()
    out = processor.process()
    # Check the output
    assert out == single_output_value, "Invalid output"
