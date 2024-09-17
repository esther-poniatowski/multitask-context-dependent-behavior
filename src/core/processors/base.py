#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.base` [module]

Classes
-------
:class:`Processor`
"""
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Tuple, Mapping, Dict, Any, Optional
import warnings

import numpy as np


class Processor(ABC):
    """
    Abstract base class for data processors.

    Class Attributes
    ----------------
    config_attrs: Tuple[str], default=()
        Names of the configuration parameters (fixed for each processor instance).
    input_attrs: Tuple[str], default=()
        Names of the required inputs to process.
    output_attrs: Tuple[str], default=()
        Names of the outputs, i.e. results of the processing.
    empty_data: Mapping[str, Any], default={}
        Mapping of attributes names to their corresponding empty instance.

    Attributes
    ----------
    _has_input: bool
        Flag indicating whether the class instance currently stores input data.
    _has_output: bool
        Flag indicating whether the class instance currently stores processed data (output results).
    _seed: Optional[int]
        Seed for random state initialization.

    Methods
    -------
    __init__
    __repr__
    _validate (optionally overridden in subclasses)
    _set_inputs (optionally overridden in subclasses)
    _process (abstract, required in subclasses)
    process
    set_seed
    set_random_state

    Notes
    -----
    Configuration vs. Processing Attributes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Processor classes operate on data of two distinct nature:

    - "Configuration parameters" define the fixed behavior of each class instance throughout its
    lifecycle (Examples: model parameters, tuning settings...). They are initialized in the
    constructor for each processor instance. Those parameters apply homogeneously on all
    subsequent calls of the main processing method.
    - "Processing data" is processed at runtime by one single call to the main method of the class
    instance (examples: input data, output result...). They are passed as arguments to the main
    processing method, stored temporarily, and reset when new data is provided. The initial state
    of those attributes is set empty in the constructor.

    Storing processing data as transient attributes has several purposes:

    - It facilitates the access to those attributes across multiple methods for different processing
      steps.
    - It maintains the link between the inputs and the outputs within the class instance if they
      need to be manipulated in the client code.
    - It allows to centralize the documentation of this data as attributes in the class docstring
      rather than as parameters in each method.

    Passing input data to a class instance
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Input data should be passed to the `process` method, which is common across all processor
    subclasses. It will be validated and stored it in the dedicated attributes.

    Random state initialization is fully handled by the base class. Subclasses do not need to
    manually define nor set any seed. A seed can be passed directly to the base `process` method as
    an extra input.

    Recovering output results
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    Output data is stored within the dedicated attributes when the `process` method is run (specific
    storage operations are performed within each subclass). The client code can retrieve target
    results by querying the associated property.

    Warning
    -------
    Intermediate attributes stored during processing should be mentioned among the output
    attributes to ensure they are reset when new data is processed.
    """

    config_attrs: Tuple[str, ...] = ()
    input_attrs: Tuple[str, ...] = ()
    output_attrs: Tuple[str, ...] = ()
    empty_data: Mapping[str, Any] = MappingProxyType({})

    def __init__(self, **config: Any):
        # Initialize configuration attributes
        for attr in self.config_attrs:
            if attr not in config:
                raise ValueError(f"Missing configuration in '{self.__class__.__name__}': '{attr}'")
            else:
                setattr(self, attr, config[attr])
        # Initialize data attributes to empty instances of expected types
        self._has_input: bool = False
        self._has_output: bool = False
        self._reset(inputs=True, outputs=True)
        # Declare the seed (set optionally afterwards, only if randomness is involved in operations)
        self._seed: Optional[int] = None

    def __repr__(self):
        config_values = {attr: getattr(self, attr) for attr in self.config_attrs}
        status = {"inputs": self._has_input, "outputs": self._has_output}
        return f"<{self.__class__.__name__}(status={status}, config={config_values}, inputs={self.input_attrs}, outputs={self.output_attrs})"

    def process(self, seed: Optional[int] = None, **input_data: Any) -> None:
        """
        Main processing method which receives input data and execute the processing logic.

        Parameters
        ----------
        input_data: Any
            Input data to process. It should be passed as keyword arguments (i.e. by name). Each
            argument name must be included in the `input_attrs` class-level attribute and its value
            must match the expected types defined in the `empty_data` class-level attribute.
        seed: Optional[int], default=None
            Seed for random state initialization to ensure reproducibility, if randomness is
            involved in the operations. If provided, it will set the random seed before the
            processing begins.

        Notes
        -----
        This main method is the entry point to pass data to process. It orchestrates the processing
        logic by calling the subclass-specific methods through a "template method" design pattern.

        Key Steps:

        - Setup: Handle seed and reset state.
        - Preprocessing: Validate input data, reset data attributes, store inputs.
        - Processing: Execute the core logic.
        - Postprocessing: Check output data.

        At the end of processing, all the output results computed by the concrete processor are
        stored among the attributes of the class instance. They can be accessed directly by querying
        the corresponding properties.

        Example
        -------
        Pass input data to the processor and retrieve the output data:

        >>> processor = ConcreteProcessor()
        >>> processor.process(input1=..., input2=..., seed=42)
        >>> output1 = processor.output1
        >>> output2 = processor.output2
        """
        if seed is not None:  # set new seed if provided
            self.seed = seed  # call property setter
        self.set_random_state()  # set random state again
        self._validate(**input_data)
        self._reset(inputs=True, outputs=True)  # reset all data and flags
        self._set_inputs(**input_data)
        self._process()  # subclass-specific logic (required)
        self._check_outputs()  # check if output data is complete

    def _validate(self, **input_data: Any) -> None:
        """
        Validate inputs passed to the `process` method.

        Optionally overridden in concrete subclasses for more specific validation steps.

        Parameters
        ----------
        input_data: Any
            Input data to process, received as keyword arguments in the `process` method.

        Raises
        ------
        ValueError
            If a required attribute is missing.
            If an invalid attribute name is provided.
        TypeError
            If the type of an attribute does not match the expected type.

        Notes
        -----
        Examples of flexible subclass-specific validation:

        - Check the structure or inner datatype for nested objects (e.g., lists, dictionaries).
        - Enforce the consistency of related inputs.
        - Set default values for optional inputs if not provided.

        Once more specific validation is performed by the subclass, the base class validation can be
        run by calling the parent method with `super()._validate(**input_data)`.
        """
        provided = set(input_data.keys())
        expected = set(self.input_attrs)
        missing = expected - provided
        unexpected = provided - expected
        if missing:
            raise ValueError(f"Missing inputs: {missing}")
        if unexpected:
            raise ValueError(f"Unexpected inputs: {unexpected}")

    def _reset(self, inputs: bool = True, outputs: bool = True) -> None:
        """
        Reset data to their empty state and update corresponding flags.

        Parameters
        ----------
        inputs, outputs: bool, default=True
            Whether to reset input or output data and flags.
        """
        # Determine which attributes to reset and which flag to update
        attrs_to_reset: Tuple[str, ...] = ()
        if inputs:
            attrs_to_reset += self.input_attrs
            self._has_input = False
        if outputs:
            attrs_to_reset += self.output_attrs
            self._has_output = False
        # Reset each attribute to its empty state
        for attr in attrs_to_reset:
            setattr(self, attr, self.empty_data[attr])

    def _set_inputs(self, **input_data: Any) -> None:
        """
        Store input data in the class instance attributes. Update the flag accordingly.

        Optionally overridden in concrete subclasses for more specific validation steps.

        Parameters
        ----------
        input_data: Any
            Input data to process, received as keyword arguments in the `process` method.
        """
        for input_name, input_value in input_data.items():
            setattr(self, input_name, input_value)
        self._has_input = True

    @abstractmethod
    def _process(self) -> None:
        """
        Orchestrate the specific operations performed by the concrete processor.

        Implementation required in concrete subclasses.

        Notes
        -----
        No input arguments are passed to this method since it operates on input data stored in the
        class instance after validation.

        No output is returned. Instead, target outputs (and intermediate results, if needed) should
        have been stored in the class instance attributes, under the names specified in the class
        attribute `output_attrs`. Thereby, those results are directly accessible by querying the
        corresponding attribute after this method is run in the main `process` method.
        """

    def _check_outputs(self) -> None:
        """
        Check that the output data is complete and raise an error if not. Set the flag accordingly.

        Raises
        ------
        ValueError
            If a required output is missing.

        Warning
        -------
        Criteria for the presence of an output:

        The content of the corresponding attribute should be different from the empty instance
        stored in the class-level attribute `empty_data`. This is necessary because the
        attributes are initialized in the constructor with empty instances of the expected types (to
        ensure type consistency) rather than by `None` values.

        Equality is checked with the `==` operator, which may not be suitable for all types of data.
        In particular, for numpy arrays, the `==` operator is element-wise and returns a boolean
        array. This output type is handled by this base class method since numpy arrays are commonly
        used in the outputs of processors. The comparison with the value of the empty instance is
        performed via the function `np.array_equal`.

        For more complex data types, the method `_check_outputs` should be overridden in the
        concrete subclass.
        """
        for attr in self.output_attrs:
            output = getattr(self, attr)
            empty = self.empty_data[attr]
            if not isinstance(output, np.ndarray):
                missing = output == empty
            else:
                missing = np.array_equal(output, empty)
            if missing:
                raise ValueError(f"Missing output: '{attr}', {output}")
        self._has_output = True

    @property
    def seed(self) -> Optional[int]:
        """
        Property for accessing the internal seed.

        Returns
        -------
        Optional[int]
            Current seed value.
        """
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        """
        Property setter for the seed.

        When the seed is modified, clear the output data and reset the `_has_output` flag.

        Parameters
        ----------
        value : int
            New seed value to set.
        """
        self._seed = value  # store the new seed in the internal attribute
        self._reset(outputs=True, inputs=False)  # reset output data and flag

    def set_random_state(self) -> None:
        """
        Set the random state for reproducibility, based on the current seed attribute.

        See Also
        --------
        :func:`np.random.seed`

        Notes
        -----
        This method is included in the base class as a utility which is commonly used in processors
        which involve random number generation.

        If no 'seed' attribute is set in the processor instance, a warning is raised and the random
        state initialization is skipped.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            warnings.warn(
                f"No 'seed' set in {self.__class__.__name__}. Skip random state initialization.",
                UserWarning,
            )
