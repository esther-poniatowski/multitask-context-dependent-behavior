"""
`core.processors` [subpackage]

Classes implementing mid-level data processors used for data transformations in the analysis.

Processors are designed to process data using a consistent framework across different subclasses.

- Each *class* represents one family of processors, corresponding to a specific operation.
  Examples: data filtering, model fitting, signal analysis.
- Each *instance* represents one specific processor with a set of configuration parameters.
  Examples: filtering spikes with a specific threshold.

Modules
-------
`core.processors.base_processor`
`core.processors.preprocess`

Examples
--------
1. Initialization: Set the configuration parameters for the processor instance.

>>> processor = ProcessorSubclass(config='value')

2. Processing: Apply the processing logic to the input data.

>>> output_data = processor.process(**input_data)

Notes
-----
Parameters - Static vs. Dynamic Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Processor classes operate on data of two distinct nature:

- "Configuration parameters" define the fixed behavior of each processor instance throughout its
  lifecycle. They apply homogeneously for any call to the main processing method (`process`). They
  are stored as attributes of the instance. Examples: model parameters, tuning settings...
- "Input/output data" is passed and retrieved at runtime to the main processing method (`process`).
  It is not stored in the instance, to ensure statelessness and avoid side effects due to
  potentially mutable objects. Examples: recorded spiking times, time stamps, labels, indices...

Interaction with a Processor:

- Configuration parameters are passed to the processor *constructor*.
- Input data is passed to the base `process` method as *named arguments*.
- Output data is directly returned as a *tuple* (for multiple results) or as a single output.

Documentation sections in the processor modules:

- Configuration parameters are documented the the attributes of the class.
- Inputs are outputs are documented in the subclass-specific main `process` method (arguments and
  returned values respectively)

Inputs and outputs are not stored among the attributes of the processor instance for several
reasons:

- Statelessness: Each processing call is independent of the previous ones, to prevent side effects
  due to mutable states.
- Decoupling and Transparency: Inputs and outputs explicitly passed and returned. This facilitates
  testing, since they are accessible at each step by isolating individual methods.
- Accessibility: The `process` method behaves like a pure function. In contrast, manipulating inputs
  and outputs in a dictionary format would be less straightforward to set and access them.

Methods
^^^^^^^
Each processor provides several types of methods:

- Utility methods:

  - Role: Implement subclass-specific computations or validation of the input data.
  - Nature: Static methods for reusability and testing (can be used directly without instantiating
    the processor).

- Main `process` method (abstract):

  - Role: Orchestrate the subclass-specific methods in order.
  - Nature: Instance method, that provides a consistent interface for all the processors.

The client code interacts with the main `process` method if it were a pure function:

- Input Handling: Inputs are passed as keyword arguments, with argument names matching the
  attributes of the specific processor.
- Output Retrieval: Outputs are returned as a tuple

If several utility methods provide distinct functionalities, the desired computation can be chosen
in the `process` method by TODO: Clarify the approach.

Randomness
^^^^^^^^^^
Random state initialization is handled by a decorator (`set_random_state`), which can be flexibly
applied on any method of a processor subclass (as soon as a seed is passed as argument).


Implementation
--------------
Each processor is implemented as a concrete subclass of the `Processor` base class.

Implementation Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^
1. Define the constructor to set the configuration parameters of the processor instance.
2. Implement the `process` method in the subclass to specify the processing logic.
3. Implement utility methods for the processing logic, if needed.
4. Use the decorator `@set_random_state` for methods involving randomness.

Flexible signature of the `process` methods - Liskov Substitution Principle (LSP):

- The signature of the base `process` method only specified keyword arguments.
- Nonetheless, concrete processors can specialize their signatures to specify the exact inputs
  require for their tasks. To do so, they should:

  - Include their required inputs as *named* arguments with a *default value* in the signature
    (since they are not required for all processors). Usually, the default value is `None`.
  - Include a catch-all `**kwargs` argument at the end of the signature to be consistent with the
    base class method.
  - If randomness is involved in the subclass' operations, include the seed as a keyword argument in
    the method signature for the `@set_random_state` decorator to work.

Handling inputs in the method body:

- The relevant inputs specified in the signature of the `process` method can be directly manipulated
  within the method body.
- To eliminate type checking errors in case of `None` default values, use an `assert` statement at
  the beginning of the method. TODO: Find an more elegant solution.
- The `kwargs` dictionary recovered within the method body contains the *remaining* inputs which are
  not isolated in the signature (usually, it is useless).

Justification of this design: The subclass specify their required inputs in the method signature for
transparency. In contrast, they could have been passed and retrieved from the `kwargs` dictionary
(e.g. using the `get` method for dictionaries), but it would have been less explicit and more
cumbersome.

Examples
^^^^^^^^
Define a processor subclass which performs a basic data transformation:

.. code-block:: python

    class ExampleProcessor(Processor):

        # --- Set the configuration parameters via the constructor

        def __init__(self, min_length: int = 1, factor: float = 1.0):
            self.min_length = min_length
            self.factor = factor

        # --- Implement the processor method - Specify signature and processing logic

        def process(self, input_arr: np.ndarray | None = None, **kwargs) -> np.ndarray:
            assert input_arr is not None # for type checking
            self.validate(min_length=self.min_length, input_arr=input_arr)
            output_arr = self.compute(input_arr=input_arr, factor=self.factor)
            return output_arr

        # --- Implement utility methods for the processing logic

        @staticmethod
        def validate(min_length: int, input_arr: np.ndarray):
            l = len(input_arr)
            if l < min_length:
                raise ValueError(f"Invalid length for input array: {l} < {min_length}")

        @staticmethod
        def compute(input_arr: np.ndarray, factor: float) -> np.ndarray:
            return input_arr * factor

Use the processor on actual input data:

.. code-block:: python

    processor = ExampleProcessor(min_length=2, factor=2.0)
    output_arr = processor.process(input_arr=np.array([1, 2, 3]))
    # Output: array([2, 4, 6])}

"""
