#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.base_processor` [module]

Classes
-------
`Processor`

Notes
-----
Each subclass of `Processor` should define a dataclass that inherits from this class.

For input validation, pass the dictionary `input_data` received by the base processor's `process`
method to the subclass-specific `_pre_process` method. Unpack the keyword arguments within the
pre-processing method.

For output validation, pass the tuple `output_data` returned by the subclass-specific `_process`
method to the subclass-specific `_post_process` method. Unpack the return values within the
post-processing method. If the processor returns a single value, format it as a tuple.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict, TypeVar, Generic, Union
import warnings

import numpy as np

I = TypeVar("I", bound=Any)
"""Type variable representing input data to the processor."""

O = TypeVar("O", bound=Any)
"""Type variable representing output data from the processor."""


class Processor(ABC, Generic[I, O]):
    """
    Abstract base class for data processors.

    Class Attributes
    ----------------
    is_random: bool, default=False
        Flag indicating whether randomness is involved in the operations of the processor.

    Attributes
    ----------
    seed: Optional[int]
        Seed for random state initialization.
    config_params: Tuple[str], default=()
        Names of the configuration parameters (fixed for each processor instance).

    Methods
    -------
    `process`
    `_pre_process`
    `_process` (abstract, required in subclasses)
    `_post_process`
    `set_random_state`

    Notes
    -----
    Configuration vs. Processing Attributes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Processor classes operate on data of two distinct nature:

    - "Configuration parameters" define the fixed behavior of each processor instance throughout its
      lifecycle. They are initialized in the constructor and apply homogeneously on any call to the
      main processing method. Examples: model parameters, tuning settings...
    - "Input data" is passed at runtime to the main processing method (`process`). By default, this
      data is not stored in the instance to ensure statelessness and avoid side effects due to
      mutable objects. Examples: recorded spiking times, time stamps, labels, indices...

    Interaction with a Processor
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Input data should be passed to the base `process` method as *named arguments*.

    Output data (target results) is directly returned as a *tuple*, as with a pure function.

    For each subclass, inputs are outputs are specified in the documentation of the
    subclass-specific main `_process` method.

    Randomness
    ^^^^^^^^^^
    Random state initialization is fully handled by the base class. Subclasses do not need to
    manually define nor set any seed. A seed can be passed directly to the base `process` method as
    an extra input.
    """

    is_random: bool = False

    def __init__(self, **config_params: Any) -> None:
        # Initialize configuration parameters
        for attr, value in config_params.items():
            setattr(self, attr, value)
        # Register configuration parameters names for logging
        self.config_params: Tuple[str, ...] = tuple(config_params.keys())
        # Declare the seed (optionally set if randomness is involved in operations)
        self.seed: Optional[int] = None

    def __repr__(self):
        config = ", ".join(f"{attr}={getattr(self, attr)}" for attr in self.config_params)
        return f"<{self.__class__.__name__}(config={config})"

    def process(self, seed: Optional[int] = None, **input_data: I) -> Union[O, Tuple[O, ...]]:
        """
        Main processing method: receives input data, execute processing operations, return results.

        Parameters
        ----------
        input_data : Any
            Input data to process, passed as keyword arguments (i.e. by name). Each argument name
            must be included among the attributes of the data class associated with the processor,
            and its value must match the expected types.
            .. _input_data:
        seed : Optional[int], default=None
            Seed for random state initialization to ensure reproducibility, if randomness is
            involved in the operations. If provided, it will set the random seed before the
            processing begins.

        Returns
        -------
        output_data : Any
            Output data computed by the processor. Single or multiple values (tuple) as defined in
            the output data class, in the order in which its attributes are defined.
            .. _output_data:

        Notes
        -----
        This main method is the entry point to pass data to process. It orchestrates the processing
        logic by calling the subclass-specific methods through a "template method" design pattern.

        Key Steps:

        - Setup: Initialize the random state from the seed, if randomness is involved.
        - Pre-processing (optional): Validate input data, set default values, and perform
          additional subclass-specific pre-processing if necessary.
        - Processing: Execute the subclass-specific operations to compute its target results.
        - Postprocessing (optional): Perform subclass-specific post-processing if necessary.

        Example
        -------
        Pass input data to the processor and retrieve the output data:

        >>> processor = ConcreteProcessor()
        >>> output1, output2 = processor.process(input1=..., input2=..., seed=42)

        Implementation
        --------------
        The methods `_pre_process`, `_process`, and `_post_process` should be consistent in the
        types of their inputs and outputs:

        - `_process` takes a dictionary and returns a tuple, to behave like the `process` method.
        - `_pre_process` takes and returns a dictionary, to insert in the pipeline while preserving
          the same format as the input data.
        - `_post_process` takes a tuple and returns a tuple, for the same reason.
        """
        if self.is_random:
            self.set_random_state(seed)
        input_data = self._pre_process(**input_data)
        output_data = self._process(**input_data)  # subclass-specific logic (required)
        if not isinstance(output_data, tuple):  # if single output: format as a tuple
            output_data = (output_data,)
        output_data = self._post_process(*output_data)
        if len(output_data) == 1:  # if single output: return as a single value (not a tuple)
            return output_data[0]
        return output_data

    def _pre_process(self, **input_data: I) -> Dict[str, I]:
        """
        Pre-processing operations. Optionally overridden in concrete processor subclasses.

        Parameters
        ----------
        input_data : Any
            See the argument :ref:`input_data`.

        Returns
        -------
        Dict[str, Any]
            Preprocessed input data, in the same format as the argument `input_data` (dictionary) to
            ensure compatibility with the `process` method pipeline.

        Notes
        -----
        Examples of subclass-specific pre-processing:

        - Check the structure or inner datatype for nested objects (e.g., lists, dictionaries).
        - Enforce the consistency of related inputs.
        - Set default values for optional inputs if not provided.
        """
        return input_data

    @abstractmethod
    def _process(self, **input_data: I) -> Union[O, Tuple[O, ...]]:
        """
        Orchestrate the specific operations performed by the concrete processor.

        Implementation required in each concrete processor subclass.

        Parameters
        ----------
        input_data : Any
            See the argument :ref:`input_data`.

        Returns
        -------
        Any
            Output data computed by the processor, in a format similar to :ref:`output_data`.
        """

    def _post_process(self, *output_data: O) -> Union[O, Tuple[O, ...]]:
        """
        Post-processing operations. Optionally overridden in concrete processor subclasses.

        Parameters
        ----------
        output_data : Any
            See the returned value :ref:`output_data`.

        Returns
        -------
        Any
            Post-processed output data, in a format similar to :ref:`output_data`.

        Notes
        -----
        Examples of subclass-specific post-processing:

        - Check the consistency of the output data.
        - Clear temporary data in case they had been stored during processing.
        - Logging.
        """
        return output_data

    def set_random_state(self, seed) -> None:
        """
        Set the random state for reproducibility, store the current seed in an attribute.

        Parameters
        ----------
        seed : Optional[int]
            See the argument :ref:`seed`.

        Notes
        -----
        This base class method serves as a utility for those concrete processors which involve
        random number generation.

        If no 'seed' attribute is set in the processor instance, a warning is raised and the random
        state initialization is skipped.

        See Also
        --------
        :func:`np.random.seed`
        """
        if seed is not None:
            self.seed = seed
            np.random.seed(self.seed)
        else:
            msg = f"No 'seed' set in {self.__class__.__name__}. Skip random state initialization."
            warnings.warn(msg, UserWarning)
