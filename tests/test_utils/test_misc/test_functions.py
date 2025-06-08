"""
:mod:`test_core.test_utils.test_functions` [module]

See Also
--------
:mod:`utils.misc.functions`: Tested module.
"""

import pytest

from utils.misc.functions import filter_kwargs

# Sample functions for testing
# ----------------------------


def function_a(arg1, arg2="default"):
    """Dummy function with positional and keyword arguments."""
    return


input_a = {"arg1": 1, "arg2": "2", "arg3": 3}
expected_a = {"arg1": 1, "arg2": "2"}


def function_b(arg1, *args, **kwargs):
    """Dummy function with positional and variable keyword arguments."""
    return


input_b = {"arg1": 1, "arg2": 2, "arg3": 3}
expected_b = {"arg1": 1}


@pytest.mark.parametrize(
    "func, kwargs, expected",
    argvalues=[(function_a, input_a, expected_a), (function_b, input_b, expected_b)],
    ids=["pos_or_kw_args", "var_args"],
)
def test_filter_kwargs(func, kwargs, expected):
    """
    Test for :func:`filter_kwargs`.

    Test Inputs
    -----------
    func: Callable
        Function to be tested.
    kwargs: dict
        Potential keyword arguments to pass to the function.

    Expected Outputs
    ----------------
    expected: dict
        Valid keyword arguments for the function.
    """
    valid_kwargs = filter_kwargs(func, **kwargs)
    assert valid_kwargs == expected, f"Expected {expected}, Got {valid_kwargs}"
