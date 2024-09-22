#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.base_processor` [module]

Classes
-------
:class:`Processor`
:class:`ProcessorInput`
:class:`ProcessorOutput`
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, asdict, astuple
from typing import Tuple, Any, Optional, TypeVar, Generic, Dict
import warnings

import numpy as np


@dataclass
class ProcessorInput:
    """
    Base class for processor inputs.

    Notes
    -----
    Each subclass of `Processor` should define a dataclass that inherits from this class.

    Usage for input validation:

    - To capture mismatches between the expected and actual inputs (missing or extra inputs), pass
      the dictionary `input_data` received by the processor's `_pre_process` method to the data
      class. Unpack the keyword arguments in the data class constructor to match the expected
      attributes.
    - (Optional) To add more specific validation, use the data class's `validate` method and call it
      in the processor's `_pre_process` method (e.g. if interaction with the processor's
      configuration parameters is required).
    """

    def validate(self, **config_params: Any):
        """
        Validate the input data.

        Parameters
        ----------
        config_params: Any
            Configuration parameters from the processor that may affect validation.
        """


@dataclass
class ProcessorOutput:
    """
    Base class for processor outputs.

    Notes
    -----
    Each subclass of `Processor` should define a specific dataclass that inherits from this class.

    Usage for output validation:

    - To capture mismatches between the expected and actual outputs (missing or extra outputs), pass
      the tuple `output_data` returned by the processor's `_process` method to the data class.
      Unpack the return values in the data class constructor to match the expected attributes in the
      order they are defined.
    - (Optional) To add more specific validation, use the data class's `__post_init__` method or the
     processor's `_post_process` method.
    """


I = TypeVar("I", bound=ProcessorInput)
"""Type variable for the input data class associated with a specific processor."""

O = TypeVar("O", bound=ProcessorOutput)
"""Type variable for the output data class associated with a specific processor."""


class Processor(ABC, Generic[I, O]):
    """
    Abstract base class for data processors.

    Class Attributes
    ----------------
    config_params: Tuple[str], default=()
        Names of the configuration parameters (fixed for each processor instance).
    input_dataclass: type[I]
        Reference to the data class which defines the expected input arguments for the processor.
    output_dataclass: type[O]
        Reference to the data class which defines the expected output arguments for the processor.
    output_names: Tuple[str], default=()
        Names of the outputs in the output dataclass, ordered as they are returned by the processor.
    is_random: bool, default=False
        Flag indicating whether randomness is involved in the operations of the processor.

    Attributes
    ----------
    seed: Optional[int]
        Seed for random state initialization.

    Methods
    -------
    process
    _pre_process
    _process (abstract, required in subclasses)
    _post_process
    set_random_state

    Notes
    -----
    Configuration vs. Processing Attributes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Processor classes operate on data of two distinct nature:

    - "Configuration parameters" define the fixed behavior of each class instance throughout its
      lifecycle (examples: model parameters, tuning settings...). They are initialized in the
      constructor for each processor instance. Those parameters apply homogeneously on any call to
      the main processing method.
    - "Processing data" is passed at runtime to the main processing method (examples: input data).
      By default, this data is not stored in the instance to ensure statelessness and avoid side
      effects due to mutable objects.

    Passing input data to a class instance
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Input data should be passed to the `process` method as named arguments.

    Recovering output results
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    At the end of processing, the target results are directly returned from the main `process`
    method, as with a pure function (tuple `output_data`). If multiple outputs are generated, their
    order is specified in the subclass-specific documentation.

    Randomness
    ^^^^^^^^^^
    Random state initialization is fully handled by the base class. Subclasses do not need to
    manually define nor set any seed. A seed can be passed directly to the base `process` method as
    an extra input.
    """

    config_params: Tuple[str, ...] = ()
    input_dataclass: type[I]
    output_dataclass: type[O]
    is_random: bool = False

    def __init__(self, **config_args: Any):
        # Initialize configuration attributes
        for attr in self.config_params:
            if attr not in config_args:
                raise ValueError(f"Missing configuration in '{self.__class__.__name__}': '{attr}'")
            else:
                setattr(self, attr, config_args[attr])
        # Declare the seed (optionally set if randomness is involved in operations)
        self.seed: Optional[int] = None

    def __repr__(self):
        config = self.get_config_params()
        input_names = [field.name for field in fields(self.input_dataclass)]
        output_names = [field.name for field in fields(self.output_dataclass)]
        return f"<{self.__class__.__name__}(config={config}, inputs={input_names}, outputs={output_names})"

    def get_config_params(self) -> Dict[str, Any]:
        """
        Get the configuration parameters as a dictionary.

        Returns
        -------
        config_params: Dict[str, Any]
            Configuration parameters of the processor instance.
        """
        return {attr: getattr(self, attr) for attr in self.config_params}

    def process(self, seed: Optional[int] = None, **input_data: Any) -> Any:
        """
        Main processing method: receives input data, execute processing operations, return results.

        Parameters
        ----------
        input_data: Any
            Input data to process, passed as keyword arguments (i.e. by name). Each argument name
            must be included among the attributes of the data class associated with the processor,
            and its value must match the expected types.
            .. _input_data:
        seed: Optional[int], default=None
            Seed for random state initialization to ensure reproducibility, if randomness is
            involved in the operations. If provided, it will set the random seed before the
            processing begins.

        Returns
        -------
        output_data: Any
            Output data computed by the processor. Single or multiple values (tuple) as defined in
            the output data class, in the order in which its attributes are defined.
            .. _output_data:

        Notes
        -----
        This main method is the entry point to pass data to process. It orchestrates the processing
        logic by calling the subclass-specific methods through a "template method" design pattern.

        Key Steps:

        - Setup: Initialize the random state from the seed, if randomness is involved.
        - Pre-processing: Validate input data (base default behavior) and perform subclass-specific
          pre-processing (optional).
        - Processing: Execute the subclass-specific operations to compute its target results.
        - Postprocessing: Validate output data (base default behavior) and perform subclass-specific
          post-processing (optional).

        Example
        -------
        Pass input data to the processor and retrieve the output data:

        >>> processor = ConcreteProcessor()
        >>> output1, output2 = processor.process(input1=..., input2=..., seed=42)

        Implementation
        --------------
        The methods `_pre_process`, `_process`, and `_post_process` should be consistent in the
        types of their inputs and outputs:

        - `_process` takes a dictionary and returns a tuple, as the `process` method.
        - `_pre_process` takes and returns a dictionary, so that it could be removed from the
          pipeline without changing the signature of the `process` and `_process` methods.
        - `_post_process` takes a tuple and returns a tuple, for the same reason.
        """
        if self.is_random:
            self.set_random_state(seed)
        input_valid = self._pre_process(**input_data)
        output_data = self._process(**input_valid)  # subclass-specific logic (required)
        if not isinstance(output_data, tuple):  # if single output: format as a tuple
            output_data = (output_data,)
        output_valid = self._post_process(*output_data)
        if len(output_valid) == 1:  # if single output: return as a single value (not a tuple)
            return output_valid[0]
        return output_valid

    def _pre_process(self, **input_data: Any) -> Dict[str, Any]:
        """
        Pre-processing operations.

        Optionally overridden in concrete processor subclasses.
        Base implementation: validate the input data passed to the `process` method.

        Parameters
        ----------
        input_data: Any
            See :ref:`input_data`.

        Returns
        -------
        input_valid: Dict[str, Any]
            Validated input data. The return type is a dictionary to ensure compatibility with the
            in the `process` method pipeline.

        Raises
        ------
        ValueError
            If an input argument has an invalid value based on the constraints enforced by the
            `ProcessorInput` data class.

        Notes
        -----
        Examples of subclass-specific pre-processing:

        - Check the structure or inner datatype for nested objects (e.g., lists, dictionaries).
        - Enforce the consistency of related inputs.
        - Set default values for optional inputs if not provided.

        See Also
        --------
        :class:`ProcessorInput`
        :meth:`dataclasses.asdict`
        """
        config_params = self.get_config_params()
        try:
            input_obj = self.input_dataclass(**input_data)  # instantiate
            input_obj.validate(**config_params)  # validate based on config params (if needed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid input to {self.__class__.__name__}: {exc}") from exc
        return asdict(input_obj)  # return validated input data as a dictionary

    @abstractmethod
    def _process(self, **input_data: Any) -> Any:
        """
        Orchestrate the specific operations performed by the concrete processor.

        Implementation required in each concrete processor subclass.

        Parameters
        ----------
        input_data: Any
            See :ref:`input_data`.

        Returns
        -------
        Any
            Output data computed by the processor, in a format similar to :ref:`output_data`.
        """

    def _post_process(self, *output_data: Any) -> Any:
        """
        Post-processing operations.

        Optionally overridden in concrete processor subclasses.
        Base implementation: validate the output data returned by the `_process` method.

        Parameters
        ----------
        output_data: Any
            See :ref:`output_data`.

        Returns
        -------
        Any
            Post-processed output data, in a format similar to :ref:`output_data`.

        Raises
        ------
        ValueError
            If the output data is invalid based on the constraints enforced by the `ProcessorOutput`
            data class.

        Notes
        -----
        Examples of subclass-specific post-processing:

        - Check the consistency of the output data.
        - Clear temporary data in case they had been stored during processing.
        - Logging.

        See Also
        --------
        :class:`ProcessorInput`
        :meth:`dataclasses.astuple`
        """
        try:
            output_obj = self.output_dataclass(*output_data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid output from {self.__class__.__name__}: {exc}") from exc
        return astuple(output_obj)

    def set_random_state(self, seed) -> None:
        """
        Set the random state for reproducibility, store the current seed in an attribute.

        Parameters
        ----------
        seed: Optional[int]
            See :attr:`seed`.

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
            warnings.warn(
                f"No 'seed' set in {self.__class__.__name__}. Skip random state initialization.",
                UserWarning,
            )
