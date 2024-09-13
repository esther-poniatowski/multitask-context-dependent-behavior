#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors.base` [module]

Classes
-------
:class:`Processor`
"""
from abc import ABCMeta, abstractmethod
from types import MappingProxyType
from typing import Tuple, Mapping, Dict, Type, Self, Any


class ProcessorMeta(ABCMeta):
    """
    Metaclass to create processor classes (subclasses of :class:`Processor`)

    This metaclass implements general logic that is common to all processor classes. Thereby, each
    processor class can remain focused on its own processing logic.

    Responsibilities:

    - Create a property for each "dynamic attribute".
    - Check the consistency of certain class-level attributes.
    - Validate the types of the dynamic attributes specified in the class-level attribute.

    Class Attributes
    ----------------
    dynamic_attributes_names: Tuple[str]
        Names of the class-level attributes which should be defined in each processor class to
        specify the names of the dynamic attributes.

    Attributes
    ----------
    input_attrs: Processor.input_attrs
    output_attrs: Processor.output_attrs
    intermediate_attrs: Processor.intermediate_attrs

    Methods
    -------
    :meth:`__new__`
    :meth:`make_property`
    :meth:`validate_types`
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

    This is performed via the method `make_property` which creates a property to access the dynamic
    attribute. The inner function `prop_getter` captures the attribute name as a default argument.
    """

    dynamic_attributes_names = ("input_attrs", "output_attrs", "intermediate_attrs")

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
        1. Get the dynamic attributes declared in the class dictionary.
        2. Define a property for each dynamic attribute. Use a closure (inner function) to capture
           the attribute name (see "Late Binding Issue").
        """
        # Get class attributes declared in the class dictionary
        dynattr2type = dct.get("dynattr2type", {})
        dynamic_attrs = sum([dct.get(attr, ()) for attr in mcs.dynamic_attributes_names])
        # Validate the types specified for the dynamic attributes
        mcs.validate_types(dynattr2type)
        # Define a property to access each dynamic attribute
        for attr in dynamic_attrs:
            dct[attr] = mcs.make_property(attr)
        # Check the consistency of the class-level attributes
        mcs.check_consistency(dct)
        # Create the new processor class
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def make_property(attr):
        """
        Create a property to access an internal dynamic attribute.

        Parameters
        ----------
        attr: str
            Name of the dynamic attribute (without leading underscore).

        Returns
        -------
        property
            Property to access the internal dynamic attribute.
        """

        def prop_getter(self):
            return self.__dict__.get(f"_{attr}")  # leading underscore

        return property(prop_getter)

    @staticmethod
    def validate_types(dynattr2type):
        """
        Validate the types of the dynamic attributes specified in the class-level attribute
        `dynattr2type`.

        Types are considered valid if they can be initialized without arguments.

        Parameters
        ----------
        dynamic_attrs: Dict[str, ...]
            Mapping of dynamic attributes to their corresponding type.

        Raises
        ------
        ValueError
            If the dynamic attributes are not defined as tuples.
        """
        for attr, tpe in dynattr2type.items():
            try:
                tpe()  # initialize without arguments
            except TypeError as exc:
                raise TypeError(
                    f"Type {tpe} for attribute '{attr}' cannot be initialized without arguments."
                ) from exc

    @staticmethod
    def check_consistency(dct):
        """
        Check that the class-level attributes are consistent with each other.

        Specifically, check that the keys in :attr:`Data.dynattr2type` (dict) match the strings in
        :attr:`Processor.input_attrs`, :attr:`Processor.output_attrs` and
        :attr:`Processor.intermediate_attrs` (tuples).

        Parameters
        ----------
        dct: Dict[str, ...]: See :meth:`__new__`.

        Implementation
        --------------
        Compare the sets of keys in the dictionary :attr:`Processor.dynattr2type` and the content of
        the tuples by converting the latter to sets.

        Raise
        -----
        ValueError
            If the keys in :attr:`Processor.dynattr2type` do not match those in the tuples which
            indicate the dynamic attributes.
        """
        set_dynattrs = set(dct["input_attrs"] + dct["output_attrs"] + dct["intermediate_attrs"])
        if set(dct["dynattr2type"].keys()) != set_dynattrs:
            raise ValueError("Inconsistent dynamic attributes")


class Processor(metaclass=ProcessorMeta):
    """
    Abstract base class for data processors.

    Class Attributes
    ----------------
    input_attrs: Tuple[str], default=()
        Names of the inputs to process.
    output_attrs: Tuple[str], default=()
        Names of the outputs, i.e. results of the processing.
    intermediate_attrs: Tuple[str], default=()
        Names of the intermediate data which should be stored for some processing steps.
    dynattr2type: Mapping[str, Type], default={}
        Mapping of dynamic attributes to their corresponding type.

    Attributes
    ----------
    _has_data: bool
        Flag indicating whether the class instance currently stores dynamic data.

    Methods
    -------
    __init__
    __repr__
    set_dynamic_attr
    set_empty_data
    process
    _validate
    _process
    check_inputs

    Notes
    -----
    Configuration vs. Dynamic Attributes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Processor classes operate on data of two distinct nature:

    - "Configuration parameters" define the fixed behavior of each class instance throughout its
      lifecycle (Examples: model parameters, tuning settings...). They are initialized in the
      constructor for each processor instance. Those parameters apply homogeneously on all
      subsequent calls of the main processing method.
    - "Dynamic data" is processed by one single call to the main method of the class instance
      (examples: input data, output result of the processing...). They are passed as arguments to
      the main processing method, stored transiently, and reset when new data is provided. The
      initial state of those attributes is set empty in the constructor. They are marked as internal
      attributes with a leading underscore, and are accessed through read-only properties (to
      prevent arbitrary modifications which would break the consistency among dependent attributes).

    Storing dynamic data as transient attributes has several purposes:

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
    dynattr2type: Mapping[str, Type] = MappingProxyType({})

    def __init__(self, **config):
        # Initialize configuration attributes
        for attr in self.config_attrs:
            if attr not in config:
                raise ValueError(f"Missing configuration attribute: '{attr}'")
            else:
                setattr(self, attr, config[attr])
        # Initialize internal attributes to empty instances of expected types
        self._has_data = False
        self.set_empty_data()

    def __repr__(self):
        config_values = {attr: getattr(self, attr) for attr in self.config_attrs}
        return f"<{self.__class__.__name__}(status={self._has_data}, config={config_values})>"

    def set_dynamic_attr(self, attr, value):
        """
        Set a value in the internal attribute to store dynamic data.

        Parameters
        ----------
        attr: str
            Name of the dynamic attribute.
        value: Any
            Value to set.

        Raises
        ------
        ValueError
            If the attribute name is invalid.
        TypeError
            If the type of the value does not match the expected type specified in `dynattr2type`.
        """
        if attr not in self.dynattr2type:
            raise ValueError(f"Invalid dynamic attribute: '{attr}'")
        tpe = self.dynattr2type[attr]  # expected type
        if not isinstance(value, tpe):
            raise TypeError(f"Invalid type: type({attr}) = {type(value)} != {tpe}")
        setattr(self, f"_{attr}", value)  # leading underscore

    def set_empty_data(self):
        """Reset the dynamic attributes to their default empty state."""
        for attr, tpe in self.dynattr2type.items():
            self.set_dynamic_attr(attr, tpe())
        self._has_data = False  # reset flag

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
        # Set input data
        self.check_inputs(**input_data)
        self.set_empty_data()
        for input_name, input_value in input_data.items():
            self.set_dynamic_attr(input_name, input_value)
        # Execute subclass-specific processing logic
        self._validate(**input_data)  # optional
        output_data = self._process(**input_data)  # required
        # Store the results and return them
        for attr, value in output_data.items():
            self.set_dynamic_attr(attr, value)
        self._has_data = True
        return output_data

    def _validate(self, **input_data) -> None:
        """
        Perform more specific validation of inputs (structure, datatype... ). To be implemented in
        concrete subclasses.

        Parameters
        ----------
        input_data: Mapping[str, Any]
            Input data to process.
        """
        pass

    @abstractmethod
    def _process(self, **input_data) -> Dict[str, Any]:
        """
        Orchestrate the specific processing logic. To be implemented in concrete subclasses.

        Returns
        -------
        output_data: Dict[str, Any]
            Processed output, target results of the operations performed by the concrete processor.
            Keys: Same as the dynamic attributes specified in the class attribute `output_attrs`.
            Values: Results of the processing.

        Notes
        -----
        The return format is a dictionary in order to automatically map the output to the internal
        attributes in the base class `process` method.
        """

    def check_inputs(self, **input_data: Mapping[str, Any]) -> None:
        """
        Check the input data passed to the processor.

        Parameters
        ----------
        input_data: Mapping[str, Any]
            Input data to process.

        Raises
        ------
        ValueError
            If an invalid input attribute is provided.
            If a required input attribute is missing.
        """
        for input_name in input_data:
            if input_name not in self.input_attrs:
                raise ValueError(f"Invalid input attribute: '{input_name}'")
        for input_name in self.input_attrs:
            if input_name not in input_data:
                raise ValueError(f"Missing input attribute: '{input_name}'")
