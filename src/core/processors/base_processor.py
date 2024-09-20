#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.base_processor` [module]

Classes
-------
:class:`Processor`

Implementation
--------------
Justification of Key Design Choices

**Dataclasses for Input and Output**

For each processor, the types of inputs and outputs are specified in two dataclasses:
`ProcessorInput` and `ProcessorOutput`. This approach has several purposes:

- Centralized documentation: Inputs and outputs are defined once with there types. They can be
  referenced by the processor's methods.
- Automatic Validation: The structure of the inputs and outputs (including number, name, default
  values) is enforced via the data classes. This ensures a seamless flow through the processing
  pipeline when chaining several methods. In contrast, a dictionary-based approach using a
  class-level attribute defined in the processor would require manual validation and potentially
  clutter the processing logic.

**Passing Inputs and Retrieving Outputs**

The client code interacts with the main `process` method if it were a pure function:

- Input Handling: Inputs are passed as keyword arguments, with argument names matching the
  attributes of the dataclass associated with the processor.
- Output Retrieval: Outputs are returned as a tuple (for multiple results) or as a single output.
  The order of the outputs corresponds to the order of the attributes in the `ProcessorOutput`
  dataclass. In contrast, a dictionary format would be less straightforward for accessing the
  results.

Inputs and outputs are not stored among the attributes of the processor instance, for several
reasons:

- Statelessness: Each processing call is independent of the previous ones, to prevent side effects
  due to mutable states.
- Decoupling and Transparency: Inputs and outputs explicitly passed and returned throughout the
  pipeline. This facilitates testing, since they are accessible at each step by isolating individual
  methods.

**Manipulating Inputs and Outputs through the Pipeline**

- Consistent formats: Within the `process` method, inputs are manipulated as a dictionary, while the
  outputs are handled as a tuple. Those formats are maintained across the `_pre_process`, `_process`
  and `_post_process` methods to ensure consistency throughout the pipeline. Thereby, modifications
  in any step does not affect the overall interface of the processor.
- Flexible signatures: In signature of the base `process` method supports any number of inputs
  passed as keyword arguments. Concrete processors's methods can specialize their signatures to
  specify the exact inputs require for their tasks.
- Unpacking: When passed to internal methods and data classes, inputs and outputs are unpacked from
  their respective formats (dictionary and tuple). This allows direct access to their content
  within methods without the need to extract them from a container.

**Template Method Pattern**

The base class provides a template method design pattern for the processing pipeline:

- Abstract `_process`method: This method must be implemented by each concrete processor subclass.
- Optional `_pre_process` and `_post_process` methods: These methods provide a basic implementation
  by default but also serve as optional hooks for subclass-specific validation or transformation
  operations.

Advantages of the template method pattern:

- Separation of Concerns: Each method focuses on a distinct phase of the pipeline, adhering to the
  Single Responsibility Principle.
- Modularity and Extensibility: Each part of the pipeline can be updated individually without
  affecting the other steps, while preserving the overall structure of the pipeline.
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

    - Pass the dictionary `input_data` received by the `Processor._pre_process` method to the data
      class to capture mismatches between the expected and actual inputs (missing or extra inputs).
      This dictionary should be passed by unpacking the keyword arguments in the data class
      constructor to match the expected attributes.
    - More specific validation can be added by adding a `__post_init__` method in the dataclass
      subclass or by implementing a subclass-specific `_pre_process` method if interaction with the
      processor's configuration parameters is required.

    Examples of flexible subclass-specific validations:

    - Check the structure or inner datatype for nested objects (e.g., lists, dictionaries).
    - Enforce the consistency of related inputs.
    - Set default values for optional inputs if not provided.

    Usage for documentation: The data class centralizes the definition of the expected inputs for a
    processor subclass. Its attributes can be referenced from the processor's methods.

    Warning
    -------
    Type validation from the type hints is not automatically enforced by the data class decorator.
    """


@dataclass
class ProcessorOutput:
    """
    Base class for processor outputs.

    Notes
    -----
    Each subclass of `Processor` should define a specific dataclass that inherits from this class.

    Usage for output validation:

    - Pass the tuple `output_data` returned by the `Processor._process` method to the data class to
      capture mismatches between the expected and actual outputs (missing or extra outputs). The
      tuple should be passed by unpacking the return values in the data class constructor to match
      the expected attributes, in the order they are defined.
    - More specific validation can be added by adding a `__post_init__` method in the subclass or by
      implementing a subclass-specific `_post_process` method if interaction with configuration
      parameters of the processor is required.

    Warning
    -------
    Type validation from the type hints is not automatically enforced by the data class decorator.
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
        config = {attr: getattr(self, attr) for attr in self.config_params}
        input_names = [field.name for field in fields(self.input_dataclass)]
        output_names = [field.name for field in fields(self.output_dataclass)]
        return f"<{self.__class__.__name__}(config={config}, inputs={input_names}, outputs={output_names})"

    def process(self, seed: Optional[int] = None, **input_data: Any) -> Any:
        """
        Main processing method: receives input data, execute processing operations, return results.

        Parameters
        ----------
        input_data: Any
            Input data to process. It should be passed as keyword arguments (i.e. by name). Each
            argument name must be included among the attributes of the data class associated with
            the processor, and its value must match the expected types.
            .. _input_data:
        seed: Optional[int], default=None
            Seed for random state initialization to ensure reproducibility, if randomness is
            involved in the operations. If provided, it will set the random seed before the
            processing begins.

        Returns
        -------
        output_data: Any
            Output data computed by the processor. Return type: single or multiple values
            corresponding to the types defined in the output data class, in the order in which those
            attributes are defined.
            .. _output_data:

        Notes
        -----
        This main method is the entry point to pass data to process. It orchestrates the processing
        logic by calling the subclass-specific methods through a "template method" design pattern.

        Key Steps:

        - Setup: Initialize the random state from the seed, if randomness is involved.
        - Pre-processing: Validate input data (base) and perform more specific pre-processing if
          implemented in the subclass.
        - Processing: Execute the subclass-specific operations to compute its target results.
        - Postprocessing: Validate output data (base) and perform more specific post-processing if
          implemented in the subclass.

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
        self._post_process(*output_data)  # optional subclass-specific post-processing
        if len(output_data) == 1:  # if single output: return as a single value (not a tuple)
            return output_data[0]
        return output_data

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
            Validated input data.

        Raises
        ------
        TypeError
            If an input argument has an invalid type.
        ValueError
            If an input argument has an invalid value based on the constraints enforced by the
            `ProcessorInput.__post_init__` data class.

        Notes
        -----
        The return type is a dictionary to ensure compatibility with the in the `process` method
        pipeline. The dictionary is then unpacked to pass the validated input data to the
        subclass-specific `_process` method which expects keyword arguments.

        See Also
        --------
        :class:`ProcessorInput`
        :meth:`dataclasses.asdict`
        """
        try:
            input_valid = self.input_dataclass(**input_data)
        except TypeError as exc:
            valid_types = {field.name: field.type for field in fields(self.input_dataclass)}
            raise TypeError(
                f"Invalid input to {self.__class__.__name__}: {exc}. Valid types: {valid_types}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"Invalid input to {self.__class__.__name__}: {exc}") from exc
        return asdict(input_valid)

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
        TypeError
            If the output data has an invalid type.

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
            outputs_valid = self.output_dataclass(*output_data)
        except TypeError as exc:
            raise TypeError(f"Invalid output from {self.__class__.__name__}: {exc}") from exc
        return astuple(outputs_valid)

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
