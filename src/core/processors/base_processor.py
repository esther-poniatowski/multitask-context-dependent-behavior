#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.base_processor` [module]

Classes
-------
`Processor`

Notes
-----
Each subclass of `Processor` should inherit from this class.
"""
from abc import ABC, abstractmethod
from functools import wraps
from typing import Set, Any, Callable, Type, Self, TypeGuard, Tuple
import warnings

import numpy as np


# --- Base Processor Class ------------------------------------------------------------------------


class Processor(ABC):
    """
    Abstract base class for data processors.

    Attributes
    ----------
    config_params : Set[str]
        Names of the configuration parameters (fixed for each processor instance).

    Methods
    -------
    `process` (abstract)

    Notes
    -----
    Implementation guidelines for specific processors: See the documentation of the
    `core.processors` package.
    """

    config_params: Set[str]  # declare for type checking

    @classmethod
    def __init_subclass__(cls: Type[Self], **kwargs: Any) -> None:
        """
        Initialize the subclass with the `config_params` attribute.

        Arguments
        ---------
        cls : Type[Processor]
            Subclass of the `Processor` abstract base class.

        Implementation
        --------------
        - Create a class attribute `config_params` (set) to register the configuration parameters.
        - Store the original `__init__` method of the subclass.
        - Define a new `__init__` method :
            - Capture the state of `self.__dict__` before calling the original `__init__`.
            - Call the original `__init__` method to set the subclass's attributes (if any).
            - Compare the new state of `self.__dict__` with the original to identify new attributes.
            - Update the `config_params` set with these new attributes.
        - Replace the original `__init__` method with the new one.
        """
        super().__init_subclass__(**kwargs)
        cls.config_params = set()
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_dict = self.__dict__.copy()
            original_init(self, *args, **kwargs)
            new_attributes = set(self.__dict__.keys()) - set(original_dict.keys())
            cls.config_params.update(new_attributes)

        setattr(cls, "__init__", new_init)

    def __repr__(self):
        config = ", ".join(f"{attr}={getattr(self, attr)}" for attr in self.config_params)
        return f"<{self.__class__.__name__}(config={config})"

    @abstractmethod
    def process(self, *args, **kwargs: Any) -> Any:
        """
        Main processing method. Implementation is required in each concrete processor subclass.

        This method orchestrates the processing steps specific to the subclass, which includes:

        - Pre-processing (optional): validating inputs, formatting, setting default values.
        - Processing: executing the specific operations performed by the concrete processor, by
          invoking utility methods or external functions.
        - Returning output data computed by the processor.

        Parameters
        ----------
        kwargs : Any
            Input data to process, passed as keyword arguments (i.e. by name).

        Returns
        -------
        output_data : Any
            Output data computed by the processor. Single or multiple values (tuple).

        Example
        -------
        Define a concrete processor subclass:

        >>> class ConcreteProcessor(Processor):
        ...     @set_random_state
        ...     def process(self, input1=None, input2=None, seed=0, **kwargs):
        ...         # implementation

        Pass input data to the processor and retrieve the output data:

        >>> processor = ConcreteProcessor()
        >>> output1, output2 = processor.process(input1=1, input2=2, seed=42)

        Notes
        -----
        Implementation guidelines: See the documentation of the `core.processors` package.
        """


# --- Utilities ------------------------------------------------------------------------------------


def set_random_state(func: Callable) -> Callable:
    """
    Decorator to set the random state for reproducibility in processor methods.

    Arguments
    ---------
    func : Callable
        Function of method to decorate.

    Returns
    -------
    wrapper : Callable
        Decorated function or method.

    Raises
    ------
    UserWarning
        If no 'seed' is set in the keyword arguments of the method.

    Notes
    -----
    This decorator is intended for methods of processor classes that involve randomness. It sets the
    seed if the latter is provided in the keyword arguments.

    Example
    -------
    Decorate a method to set the random state:

    >>> class SubProcessor:
    ...     @set_random_state
    ...     def process(self, seed=0):
    ...         pass

    See Also
    --------
    :func:`np.random.seed`
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
            msg = f"No 'seed' set in {func.__name__}. Skip random state initialization."
            warnings.warn(msg, UserWarning)
        return func(self, *args, **kwargs)

    return wrapper
