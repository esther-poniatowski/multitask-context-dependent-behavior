#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.base` [module]

Classes
-------
:class:`ProcessorMeta`
:class:`Processor`
"""
from abc import ABCMeta, abstractmethod
from types import MappingProxyType
from typing import Tuple, Mapping, Dict, Type, Any, Optional
import warnings

import numpy as np


class ProcessorMeta(ABCMeta):
    """
    Metaclass to create processor classes (subclasses of :class:`Processor`)

    This metaclass implements general logic that is common to all processor classes. Thereby, each
    processor class can remain focused on its own processing logic.

    Responsibilities:

    - Check the consistency of certain class-level attributes.
    - Validate the types of the processing attributes specified in the class-level attribute.

    Attributes
    ----------
    input_attrs: Processor.input_attrs
    optional_attrs: Processor.optional_attrs
    output_attrs: Processor.output_attrs
    proc_data_empty: Processor.proc_data_empty
    proc_data_type: Processor.proc_data_type

    Methods
    -------
    :meth:`__new__`
    :meth:`infer_types`
    :meth:`check_consistency`

    See Also
    --------
    :class:`abc.ABCMeta`: Metaclass for *abstract base classes*.

    Warning
    -------
    In a metaclass, the instances are classes themselves (here, subclasses of :class:`Processor`).
    """

    def __new__(mcs, name, bases, dct):
        """
        Create a subclass of :class:`Processor` after ensuring its consistency.

        Parameters
        ----------
        dct: Dict[str, ...]
            Dictionary of class attributes. Initially, it contains the attributes defined in the
            body of the class being created by the metaclass. After the operations performed by the
            metaclass, it is updated with new attributes.

        Returns
        -------
        Type
            New class (subclass of :class:`Processor`).

        See Also
        --------
        :meth:`super().__new__`: Call the parent class constructor, here :class:`ABCMeta`.

        Implementation
        --------------
        1. Get the relevant attributes names declared in the class dictionary `dct`.
        2. Assign data types automatically based on the default empty instances, creating a new
           class-level attribute `proc_data_type` (stored in the class dictionary `dct`).
        """
        # Get class attributes declared in the class dictionary
        proc_data_empty = dct.get("proc_data_empty", {})
        proc_data = (
            dct.get("input_attrs", ()) + dct.get("optional_attrs", ()) + dct.get("output_attrs", ())
        )
        # Assign data types automatically based on the default empty instances
        proc_data_type = mcs.infer_types(proc_data_empty)
        dct["proc_data_type"] = proc_data_type
        # Check the consistency of the class-level attributes
        mcs.check_consistency(proc_data, proc_data_type)
        # Create the new processor class
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def infer_types(proc_data_empty: Mapping[str, Any]) -> Mapping[str, Type]:
        """
        Infer the types of the processing attributes from the default empty instances.

        Parameters
        ----------
        proc_data_empty: Mapping[str, Any]
            Mapping of attributes names to their corresponding empty instance.

        Returns
        -------
        proc_data_type: Mapping[str, Type]
            Mapping of attributes names to their corresponding type.
        """
        return MappingProxyType({attr: type(value) for attr, value in proc_data_empty.items()})

    @staticmethod
    def check_consistency(proc_data: Tuple[str, ...], proc_data_type: Mapping[str, Type]) -> None:
        """
        Check that the class-level attributes are consistent with each other.

        Specifically, check that the keys in :attr:`Data.proc_data_type` (dict) match the strings in
        :attr:`Processor.input_attrs` and :attr:`Processor.output_attrs` (tuples).

        Parameters
        ----------
        proc_data: Tuple[str, ...]
            Names of the attributes storing processing data.
        proc_data_type: Mapping[str, Type]
            Mapping of attributes names to their corresponding type.

        Implementation
        --------------
        Compare the sets of keys in the mapping :attr:`Processor.proc_data_type` and the content of
        the tuples by converting the latter to sets.

        Raise
        -----
        ValueError
            If the keys in :attr:`Processor.proc_data_type` do not match those in the tuples which
            indicate the names of the attributes storing processing data.
        """
        missing_in_types = set(proc_data) - set(proc_data_type.keys())
        if missing_in_types:
            raise ValueError(f"Missing types: {missing_in_types}")
        missing_in_data = set(proc_data_type.keys()) - set(proc_data)
        if missing_in_data:
            raise ValueError(f"Missing data: {missing_in_data}")


class Processor(metaclass=ProcessorMeta):
    """
    Abstract base class for data processors.

    Class Attributes
    ----------------
    config_attrs: Tuple[str], default=()
        Names of the configuration parameters (fixed for each processor instance).
    input_attrs: Tuple[str], default=()
        Names of the required inputs to process.
    optional_attrs: Tuple[str], default=()
        Names of the optional inputs to process.
    output_attrs: Tuple[str], default=()
        Names of the outputs, i.e. results of the processing.
    proc_data_type: Mapping[str, Type], default={}
        Mapping of attributes names to their corresponding type. Automatically inferred from the
        default empty instances in `proc_data_empty` by the metaclass.
    proc_data_empty: Mapping[str, Any], default={}
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
    _validate_data
    _validate (to be implemented in concrete subclasses, optional)
    _check_missing
    _set_data
    _process (to be implemented in concrete subclasses, required)
    process
    set_seed

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
    """

    config_attrs: Tuple[str, ...] = ()
    input_attrs: Tuple[str, ...] = ()
    optional_attrs: Tuple[str, ...] = ()
    output_attrs: Tuple[str, ...] = ()
    proc_data_empty: Mapping[str, Any] = MappingProxyType({})
    proc_data_type: Mapping[str, Type] = MappingProxyType({})

    def __init__(self, **config: Any):
        # Initialize configuration attributes
        for attr in self.config_attrs:
            if attr not in config:
                raise ValueError(f"Missing configuration in '{self.__class__.__name__}': '{attr}'")
            else:
                setattr(self, attr, config[attr])
        # Initialize data attributes to empty instances of expected types
        for attr, value in self.proc_data_empty.items():
            setattr(self, attr, value)
        # Set internal flags reflecting the current state of the processor
        self._has_input: bool = False
        self._has_output: bool = False
        # Declare the seed (set optionally afterwards, only if randomness is involved in operations)
        self._seed: Optional[int] = None

    def __repr__(self):
        config_values = {attr: getattr(self, attr) for attr in self.config_attrs}
        status = {"inputs": self._has_input, "outputs": self._has_output}
        return f"<{self.__class__.__name__}(status={status}, config={config_values})>"

    def process(self, seed: Optional[int] = None, **input_data: Any) -> None:
        """
        Main processing method which receives input data and execute the processing logic.

        Parameters
        ----------
        input_data: Any
            Input data to process. It should be passed as keyword arguments (i.e. by name).
            Each argument name must be included in the `input_attrs` class-level attribute
            and its value must match the expected types defined in the `proc_data_type`
            class-level attribute.
            Optionally, a 'seed' parameter can be passed to control random state initialization.
        seed: Optional[int], default=None
            Seed for random state initialization to ensure reproducibility, if randomness is
            involved in the operations. If provided, it will set the random seed before the
            processing begins.

        Notes
        -----
        This main method is the entry point to pass data to process. It orchestrates the processing
        logic by calling the subclass-specific methods through a "template method" design pattern.

        At the end of processing, all the output results computed by the concrete processor are
        stored among the attributes of the class instance. They can be accessed directly by querying
        the corresponding properties.

        Example
        -------
        Pass input data to the processor and retrieve the output data:

        >>> processor = ConcreteProcessor()
        >>> processor.process(input1=..., input2=..., seed=42)
        >>> output1 = processor.output1
        """
        # Set the seed if provided
        if seed is not None:
            self.seed = seed  # call the setter
        self.set_random_state()  # set the random state for reproducibility
        # Validate inputs and compute default values if necessary
        self._validate(**input_data)  # subclass-specific or default validation
        input_data = self._default(**input_data)
        # Reset data after complete validation
        for attr, value in self.proc_data_empty.items():
            setattr(self, attr, value)
        self._has_input = False  # update flag
        self._has_output = False  # reset flag
        # Store new inputs
        for input_name, input_value in input_data.items():
            setattr(self, input_name, input_value)
        self._has_input = True
        # Process
        output_data = self._process()  # subclass-specific logic (required)
        self._check_missing(output_data, self.output_attrs)
        # Store outputs after validation
        for output_name, output_value in output_data.items():
            setattr(self, output_name, output_value)
        self._has_output = True  # update flag

    def _validate(self, **input_data: Any) -> None:
        """
        Validate inputs passed to the `process` method. To be overridden in concrete subclasses for
        more specific validation steps if necessary.

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
        Default checks performed by this base method:

        - Check the presence of all required inputs specified in the class attribute `input_attrs`.
        - Check the name and type of each input against the expected type specified in the class
          attribute `proc_data_type`.

        Examples of flexible subclass-specific validation:

        - Enforce the consistency of related inputs.
        - Check the structure or inner datatype for nested objects (e.g., lists, dictionaries).

        For partial checks targeting only a subset of inputs, a subclass can still use the base
        methods involved here (`_check_missing`, `_check_name_type`) but with a restricted set of
        inputs passed as arguments or in the loop.
        """
        self._check_missing(input_data, self.input_attrs)  # required inputs
        for input_name, input_value in input_data.items():  # required and optional inputs
            if input_name in self.input_attrs or input_name in self.optional_attrs:
                self._check_name_type(input_name, input_value)
            else:
                raise ValueError(f"Unexpected input: '{input_name}'")

    def _check_name_type(self, name: str, value: Any) -> None:
        """
        Validate the name and type of a runtime processing data.

        Parameters
        ----------
        name: str
            Name of the attribute to store one piece of processing data.
        value: Any
            Value to validate.

        Raises
        ------
        ValueError
            If the attribute name is invalid.
        TypeError
            If the type of the value does not match the expected type specified in `proc_data_type`.
        """
        if name not in self.proc_data_type:
            raise ValueError(f"Invalid name: '{name}'")
        tpe = self.proc_data_type[name]  # expected type
        if not isinstance(value, tpe):
            raise TypeError(f"Invalid type: type({name}) = {type(value)} != {tpe}")

    def _check_missing(self, data: Mapping[str, Any], expected: Tuple[str, ...]) -> None:
        """
        Check missing inputs or outputs provided or recovered by the processor.

        Parameters
        ----------
        data: Mapping[str, Any]
            Data to check (either inputs or outputs).
        expected: Tuple[str, ...]
            Expected attribute names (e.g., `input_attrs` or `output_attrs`).

        Raises
        ------
        ValueError
            If a required attribute is missing or if an invalid attribute name is provided.
        """
        for attr in expected:
            if attr not in data:
                raise ValueError(f"Missing data: '{attr}'")

    def _default(self, **input_data: Any) -> Dict[str, Any]:
        """
        Compute default values for optional inputs which depend on other inputs. To be implemented
        in concrete subclasses if necessary.

        Parameters
        ----------
        input_data: Any
            Input data to process, received as keyword arguments in the `process` method.

        Returns
        -------
        input_data: Dict[str, Any]
            Input data with updated default values for optional inputs if necessary.

        Notes
        -----
        Modify the `input_data` dictionary directly or return a modified copy. This dictionary is
        used by the base class `process` method to set the final values for input data after the
        validation steps and before running the processing operations.
        """
        return input_data

    @abstractmethod
    def _process(self) -> Dict[str, Any]:
        """
        Orchestrate the specific processing logic. To be implemented in concrete subclasses.

        Returns
        -------
        output_data: Dict[str, Any]
            Processed output, target results of the operations performed by the concrete processor.
            Keys: Names specified in the class attribute `output_attrs`.
            Values: Results of the processing.

        Notes
        -----
        No input arguments are passed to this method since it operates on input data stored in the
        class instance after validation.

        No output is returned. Instead, any output data should have been stored in the class
        instance attributes so that it can be accessed directly by querying the corresponding
        attribute.
        """

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
        for output_attr in self.output_attrs:  # reset each output to its empty state
            setattr(self, output_attr, self.proc_data_empty[output_attr])
        self._has_output = False  # reset flag

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
