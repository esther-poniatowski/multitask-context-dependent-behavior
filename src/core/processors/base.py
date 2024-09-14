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
from typing import Tuple, Mapping, Dict, Type, Any


class ProcessorMeta(ABCMeta):
    """
    Metaclass to create processor classes (subclasses of :class:`Processor`)

    This metaclass implements general logic that is common to all processor classes. Thereby, each
    processor class can remain focused on its own processing logic.

    Responsibilities:

    - Create a property for each "processing attribute".
    - Check the consistency of certain class-level attributes.
    - Validate the types of the processing attributes specified in the class-level attribute.

    Attributes
    ----------
    input_attrs: Processor.input_attrs
    output_attrs: Processor.output_attrs
    intermediate_attrs: Processor.intermediate_attrs
    proc_data_empty: Processor.proc_data_empty
    proc_data_type: Processor.proc_data_type

    Methods
    -------
    :meth:`__new__`
    :meth:`infer_types`
    :meth:`make_property`
    :meth:`check_consistency`

    See Also
    --------
    :class:`abc.ABCMeta`: Metaclass for *abstract base classes*.

    Warning
    -------
    In a metaclass, the instances are classes themselves (here, subclasses of :class:`Processor`).

    Notes
    -----
    Late Binding Issue
    ^^^^^^^^^^^^^^^^^^
    In programming, the term "closure" refers to a function that captures the environment in which
    it is defined, which allows it to access the variables in the global context. For instance, the
    term "closure" can apply to an inner function which is defined within an outer function or
    within a loop.

    In the "late binding" issue, the word "binding" refers to the association between a *variable*
    and its *value*. The term "late" indicates that when a variable is used in a closure, its value
    is determined when this closure is called is *called*, not when it is *defined*. This can lead
    to unexpected behavior if the variable is updated in the outer context.

    For instance, if closures are defined within a loop and all use a common variable name, the
    value of this variable will be shared among all the closures. As a result, all the closures
    refer to the last value of the variable set in the loop.

    To capture the value of the variable in each closure at the time of its definition, the value of
    the variable can be passed as a *default argument* to the closure.

    This is performed via the method `make_property` which creates a property to access the
    processing attribute. The inner function `prop_getter` captures the attribute name as a default
    argument.
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
        1. Get the attributes names to store processing data declared in the class dictionary.
        2. Define a property to access each internal attribute. Use a closure (inner function) to
           capture the attribute name (see "Late Binding Issue").
        """
        # Get class attributes declared in the class dictionary
        proc_data_empty = dct.get("proc_data_empty", {})
        proc_data = (
            dct.get("input_attrs", ())
            + dct.get("output_attrs", ())
            + dct.get("intermediate_attrs", ())
        )
        # Assign data types automatically based on the default empty instances
        proc_data_type = mcs.infer_types(proc_data_empty)
        dct["proc_data_type"] = proc_data_type
        # Define a property to access each internal attribute
        for attr in proc_data:
            dct[attr] = mcs.make_property(attr)
        # Check the consistency of the class-level attributes
        mcs.check_consistency(proc_data, proc_data_type)
        # Create the new processor class
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def infer_types(proc_data_empty):
        """
        Infer the types of the processing attributes from the default empty instances.

        Parameters
        ----------
        proc_data_empty: Dict[str, Any]
            Mapping of attributes names to their corresponding empty instance.

        Returns
        -------
        proc_data_type: Dict[str, Type]
            Mapping of attributes names to their corresponding type.
        """
        return {attr: type(value) for attr, value in proc_data_empty.items()}

    @staticmethod
    def make_property(attr):
        """
        Create a property to access an internal attribute storing processing data.

        Parameters
        ----------
        attr: str
            Name of the future internal attribute (without leading underscore).

        Returns
        -------
        property
            Property to access the internal attribute.
        """

        def prop_getter(self):
            return self.__dict__.get(f"_{attr}")  # leading underscore

        return property(prop_getter)

    @staticmethod
    def check_consistency(proc_data, proc_data_type):
        """
        Check that the class-level attributes are consistent with each other.

        Specifically, check that the keys in :attr:`Data.proc_data_type` (dict) match the strings in
        :attr:`Processor.input_attrs`, :attr:`Processor.output_attrs` and
        :attr:`Processor.intermediate_attrs` (tuples).

        Parameters
        ----------
        proc_data: Tuple[str, ...]
            Names of the attributes storing processing data.
        proc_data_type: Dict[str, Type]
            Mapping of attributes names to their corresponding type.

        Implementation
        --------------
        Compare the sets of keys in the dictionary :attr:`Processor.proc_data_type` and the content
        of the tuples by converting the latter to sets.

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
        Names of the inputs to process.
    output_attrs: Tuple[str], default=()
        Names of the outputs, i.e. results of the processing.
    intermediate_attrs: Tuple[str], default=()
        Names of the intermediate data which should be stored for some processing steps.
    proc_data_type: Mapping[str, Type], default={}
        Mapping of attributes names to their corresponding type. Automatically inferred from the
        default empty instances in `proc_data_empty` by the metaclass.
    proc_data_empty: Mapping[str, Any], default={}
        Mapping of attributes names to their corresponding empty instance.

    Attributes
    ----------
    _has_data: bool
        Flag indicating whether the class instance currently stores processing data.

    Methods
    -------
    __init__
    __repr__
    _validate_data
    _validate_inputs (to be implemented in concrete subclasses, optional)
    _check_missing_data
    _set_data
    _process (to be implemented in concrete subclasses, required)
    process

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
      of those attributes is set empty in the constructor. They are marked as internal attributes
      with a leading underscore, and are accessed through read-only properties (to prevent arbitrary
      modifications which would break the consistency among dependent attributes).

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
    subclasses. It will be validated and stored it in the dedicated internal attributes.

    Recovering output results
    ^^^^^^^^^^^^^^^^^^^^^^^^^
    Output data is both stored within the dedicated internal attributes and returned by the
    `process` method. Thereby, the client code has more flexibility to retrieve the target results.
    It can either access the results immediately when calling the `process` method, or access them
    later by querying the associated property.
    """

    config_attrs: Tuple[str, ...] = ()
    input_attrs: Tuple[str, ...] = ()
    output_attrs: Tuple[str, ...] = ()
    intermediate_attrs: Tuple[str, ...] = ()
    proc_data_empty: Mapping[str, Any] = MappingProxyType({})
    proc_data_type: Mapping[str, Type] = MappingProxyType({})

    def __init__(self, **config):
        # Initialize configuration attributes
        for attr in self.config_attrs:
            if attr not in config:
                raise ValueError(f"Missing configuration attribute: '{attr}'")
            else:
                setattr(self, attr, config[attr])
        # Initialize internal attributes to empty instances of expected types
        for attr, value in self.proc_data_empty.items():
            self._set_data(attr, value)
        self._has_data = False  # set the flag to indicate

    def __repr__(self):
        config_values = {attr: getattr(self, attr) for attr in self.config_attrs}
        return f"<{self.__class__.__name__}(status={self._has_data}, config={config_values})>"

    def _validate_data(self, name: str, value: Any) -> None:
        """
        Validate the name and value of a runtime processing data.

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

    def _validate_inputs(self, **input_data) -> None:
        """
        Perform sub-class specific validation of inputs (structure, datatype... ). To be implemented
        in concrete subclasses.

        Parameters
        ----------
        input_data: Mapping[str, Any]
            Input data to process.
        """

    def _check_missing_data(self, data: Mapping[str, Any], expected: Tuple[str, ...]) -> None:
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

    def _set_data(self, attr, value):
        """
        Set the value of an attribute containing runtime processing data.

        Parameters
        ----------
        attr: str
            Name of the internal attribute (without leading underscore).
        value: Any
            Value to set.
        """
        setattr(self, f"_{attr}", value)  # leading underscore

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
        The return format is a dictionary in order to automatically map the output to the internal
        attributes in the base class `process` method.
        """

    def process(self, **input_data) -> Dict[str, Any]:
        """
        Main processing method which receives input data and execute the processing logic.

        Parameters
        ----------
        input_data: Mapping[str, Any]
            Input data to process.

        Raises
        ------
        TypeError
            If the input data does not match the expected types.
        ValueError
            If invalid input attribute names are provided.

        Returns
        -------
        output_data: Dict[str, Any]
            Processed output, target results of the operations performed by the concrete processor.

        Notes
        -----
        This main method orchestrates the processing logic by calling the subclass-specific methods
        through a "template method" design pattern.
        """
        # Validate inputs
        self._check_missing_data(input_data, self.input_attrs)
        for input_name, input_value in input_data.items():
            self._validate_data(input_name, input_value)
        self._validate_inputs(**input_data)  # subclass-specific validation (optional)
        # Reset data after complete validation
        for attr, value in self.proc_data_empty.items():
            self._set_data(attr, value)
        self._has_data = False  # reset flag
        # Store new inputs
        for input_name, input_value in input_data.items():
            self._set_data(input_name, input_value)
        # Process
        output_data = self._process(**input_data)  # subclass-specific logic (required)
        self._check_missing_data(output_data, self.output_attrs)
        # Store outputs
        for output_name, output_value in output_data.items():
            self._set_data(output_name, output_value)
        self._has_data = True  # update flag
        return output_data
