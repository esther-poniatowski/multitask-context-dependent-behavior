#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.utils.functions` [module]

Utilities to manipulate functions.

Functions
---------
:func:`filter_kwargs`

See Also
--------
:mod:`tests_core.test_utils.test_functions`: Unit tests for this module.
"""

import inspect
from typing import Callable


def filter_kwargs(func: Callable, **kwargs: dict) -> dict:
    """
    Retain keyword arguments which are valid for one function.

    Parameters
    ----------
    func : Callable
        Function to call.
    **kwargs : dict
        Potential keyword arguments to pass to the function.

    Returns
    -------
    dict
        Valid keyword arguments for the function.

    Warnings
    --------
    This process retains only the keyword arguments which
    match the *explicit* parameters in the function signature.
        It does not directly handle variable positional arguments (*args)
        nor variable keyword arguments (**kwargs).

    Examples
    --------
    .. code-block:: python

        def example_function(arg1, arg2, *args, kwarg1='default', **kwargs):
            print(f'arg1: {arg1}')
            print(f'arg2: {arg2}')
            print(f'args: {args}')
            print(f'kwarg1: {kwarg1}')
            print(f'kwargs: {kwargs}')

        valid_kwargs = filter_kwargs(example_function,
                                    arg1=1, arg2=2,
                                    kwarg1='value',
                                    extra_kwarg1='extra1',
                                    extra_kwarg2='extra2')
        print(valid_kwargs)
        # Output: {'arg1': 1, 'arg2': 2, 'kwarg1': 'value'}

    Notes
    -----
    This function is useful when a main function calls multiple inner functions.
    The main function receives keyword arguments and should distribute them to the inner functions.
    If invalid keyword arguments were passed to the inner functions,
    a TypeError would be raised when calling them.
    """
    signature = inspect.signature(func)
    valid_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return valid_kwargs
