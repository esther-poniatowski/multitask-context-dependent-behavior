"""
`test_core.test_processors.test_base_processor` [module]

See Also
--------
`core.processors.base_processor`: Tested module.

See Also
--------
:meth:`request.getfixturevalue`
    Access to fixtures in the current module. Here, it is necessary to store the results of the
    fixtures instead of calling them directly.
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

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from core.processors.base_processor import Processor


# --- Fixtures for a simple processor --------------------------------------------------------------


def computation(input_arr: np.ndarray) -> np.ndarray:
    """
    Simple computation performed by a processor: multiply by 2.

    Parameters
    ----------
    input_arr : np.ndarray

    Returns
    -------
    output_arr : np.ndarray
    """
    return input_arr * 2


@pytest.fixture
def subclass():
    """
    Fixture - Define a processor class implementing a simple computation.

    Returns
    -------
    TestClass : class
        Concrete subclass inheriting from the `Processor` base class.

    Notes
    -----
    To test the main functionalities of the `Processor` class:

    - Set the flag `IS_RANDOM` to `True` to test random state initialization.
    - Initialize with a single parameter `param`, which becomes the unique configuration parameter.
      Call the parent constructor and pass it as a named argument. Give a default value to
      facilitate tests.
    - Implement the abstract method `_process`, which applies the basic computation defined in the
      fixture `computation`. In this method, recover the inputs by unpacking the keyword arguments.
    - Do not override the optional methods  `_pre_process` and `_post_process` in order to test
      their base implementation, which involves seamless flow of the input and output data across
      the methods of the `process` base class method.
    """

    class TestClass(Processor):
        IS_RANDOM = True

        def __init__(self, param: int = 1):
            super().__init__(param=param)

        def _process(self, **kwargs):
            input1 = kwargs["input1"]
            input2 = kwargs["input2"]
            output1 = computation(input1)
            output2 = computation(input2)
            return output1, output2

    return TestClass


@pytest.fixture
def input_data():
    """
    Fixture - Generate valid input data matching the input data class.

    Returns
    -------
    input_data: Dict[str, np.ndarray]
        Input data to pass to the processor.
        Keys: Names of the keyword arguments recovered in the `TestClass` class.
        Values: Arrays of shape (3,) with increasing values.
    """
    keys = ["input1", "input2"]
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
def output_data(input_data):
    """
    Fixture - Generate expected output data corresponding to the input data (multiply by 2)

    Returns
    -------
    expected_output: Tuple[np.ndarray]
        Expected output from the processor after applying the processing logic.
    """
    return tuple([computation(value) for key, value in input_data.items()])


# --- Tests for with a simple processor ------------------------------------------------------------


def test_init_config(subclass):
    """
    Test the initialization of the configuration parameters by the base class constructor when
    passing named arguments.

    Test Inputs
    -----------
    subclass:
        Concrete subclass inheriting from the `Processor` base class.

    Expected Output
    ---------------
    Input and output properties should be accessible via the dynamic attribute names.
    """
    param = 42
    processor = subclass(param=param)
    assert hasattr(processor, "param"), "Missing config attribute: 'param'"
    assert processor.param == param
    assert processor.seed is None


def test_pre_process(subclass, input_data):
    """
    Test the `_pre_process` base method for seamless flow of input data.

    Test Inputs
    -----------
    input_data: Dict[str, np.ndarray]
        Complete and valid input data passed to the method by unpacking the dictionary.

    Expected Output
    ---------------
    The method should return a dictionary identical to the input data.
    """
    processor = subclass()
    recovered_inputs = processor._pre_process(**input_data)
    assert isinstance(recovered_inputs, dict)
    for key, value in input_data.items():
        assert_array_equal(recovered_inputs[key], value), "Invalid input"


def test_post_process(subclass, output_data):
    """
    Test the `_post_process` base method for seamless flow of output data.

    Test Inputs
    -----------
    output_data: Tuple[np.ndarray]
        Expected output data passed to the method by unpacking the tuple.

    Expected Output
    ---------------
    The method should return a tuple with the output data, identical to is argument.
    """
    processor = subclass()
    recovered_outputs = processor._post_process(*output_data)
    assert isinstance(recovered_outputs, tuple)
    for actual, expected in zip(recovered_outputs, output_data):
        assert_array_equal(actual, expected), "Invalid output"


def test_process(subclass, input_data, output_data):
    """
    Test the `_process` method implemented by the subclass.

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


def test_process_base(subclass, input_data, output_data):
    """
    Test the `process` method inherited from the base class.

    Test Inputs
    -----------
    subclass:
        Class inheriting from the `Processor` base class.
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
    Test the method `set_random_state` for setting the random state.

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


# --- Fixtures for a processor to test edge cases --------------------------------------------------


@pytest.fixture
def subclass_edge_cases():
    """
    Fixture - Define a processor class to test edge cases:

    - Default input argument (numpy array).
    - Single output value.

    Returns
    -------
    TestClassEdgeCases
    """

    class TestClassEdgeCases(Processor):

        def _process(self, input1=np.array([1, 2, 3]), **kwargs):
            output1 = computation(input1)
            return output1

    return TestClassEdgeCases


# --- Tests edge cases -----------------------------------------------------------------------------


def test_process_with_single_input(subclass_edge_cases):
    """
    Test the processor for edge cases.

    - No inputs is passed to the `process` method to test whether the default value is used after
      the `_pre_process` method is called in the pipeline.
    - The outputs should flow through the `_post_process` method although there is a single
      value, since it should have been converted into a tuple before calling the `_post_process`
      method. However, the return value should not be a tuple anymore since the single value should
      have been extracted from the tuple after the `_post_process` method.
    """
    # Call the processor without any input
    processor = subclass_edge_cases()
    out = processor.process()
    # Check the output
    assert isinstance(out, np.ndarray), "Output not a numpy array"
