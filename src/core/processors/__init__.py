#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors` [subpackage]

Classes implementing mid-level data processors used for data transformations in the analysis.

Modules
-------
`core.processors.base_processor`
`core.processors.preprocess`

Warning
-------
Convention for the documentation sections in the processor modules:

- Configuration Attributes: Configuration parameters of the processor, passed to the *constructor*.
- Processing Arguments: Input data to process, passed to the `process` method (base class).
- Returns: Output data after processing, returned by the `process` method (base class).

Notes
-----
**Content and Functionalities of Processors**

Processors are designed to process data using a consistent framework across different subclasses.

- Each *class* represents one family of processors, corresponding to a specific transformation.
  Examples: data filtering, model fitting, signal analysis.
- Each *instance* represents one specific processor with a set of configuration parameters.
  Examples: filtering spikes with a specific threshold.

Each processor instance operates with:

- Configuration parameters
- Input and output (dynamic data) for each specific data transformation.
- Methods for validating and processing data.

**Processing Workflow**

The core of the processing workflow is managed by the main method `process`, which is called to
execute the processing on specific inputs.

Processors are created and used as follows:

+-----------+--------------------------------------+--------------------------------------+
| Step      | Set a Processor Instance             | Process Actual Data                  |
+===========+======================================+======================================+
| Approach  | ``ProcessorSubclass(config=...)``    | ``processor.process(**input_data)``  |
+-----------+--------------------------------------+--------------------------------------+
| Use Cases | Instantiate with config settings to  | Apply processing logic to input data,|
|           | define the behavior of the processor | return target results                |
|           | for its lifetime                     |                                      |
+-----------+--------------------------------------+--------------------------------------+

**Defining Specific Processors**

Each specific processor is implemented as a concrete subclass of the base class :class:`Processor`.

Each concrete subclass define its own processing logic in three methods (template method pattern):

- `_pre_process` (optional), which is called before processing (e.g. to validate input data).
-  `_process` (required), which performs the actual processing steps.
- `_post_process` (optional), which is called after processing (e.g. to format the output).

All those methods can call other internal helper methods corresponding to different processing
steps.

Examples
--------
Define a processor subclass which performs a basic data transformation:

.. code-block:: python

    class ExampleProcessorInputs(ProcessorInput):
        input_arr: np.ndarray = np.empty(0)

        def validate(self, min_length: int = 0, **config_params):
            l = len(self.input_arr)
            if l < min_length:
                raise ValueError(f"Invalid length for input array: {l} < {min_length}")


    class ExampleProcessorOutputs(ProcessorOutput):
        output_arr: np.ndarray = np.empty(0)


    class ExampleProcessor(Processor):

        # --- Define processor class-level attributes

        config_params = ("min_length", "factor") # fixed for the processor instance
        input_dataclass = ExampleProcessorInputs
        output_dataclass = ExampleProcessorOutputs
        is_random = True

        # --- Set the configuration parameters via the parent constructor

        def __init__(self, min_length: int = 1, factor: float = 1.0):
            super().__init__(min_length=min_length, factor=factor)

        # --- Define processor methods - Specify signature and processing logic

        def _process(self, input_arr: np.ndarray = ExampleProcessorInputs.input_arr) -> np.ndarray:
            output_arr = self.input_arr * self.factor
            return output_arr


Use the processor on actual input data:

.. code-block:: python

    processor = ExampleProcessor(example_param=3)
    output_arr = processor.process(input_data=np.array([1, 2, 3]))
    # Output: array([2, 4, 6])}


Implementation
--------------
Justification of Key Design Choices

**Dataclasses for Input and Output**

For each processor, the types of inputs and outputs are specified in two dataclasses:
`ProcessorInput` and `ProcessorOutput`. This approach has several purposes:

- Centralized documentation: Inputs and outputs are defined once with there types. They can be
  referenced by the processor's methods.
- Automatic Validation: The structure of the inputs and outputs (i.e. number and name) is enforced
  by the data classes instantiation. This ensures a seamless flow through the processing pipeline
  when chaining several methods. In contrast, a dictionary-based approach using a class-level
  attribute defined in the processor would require manual validation and potentially clutter the
  processing logic.
- Delegated Validation via Dependency Injection : Setting default values or checking inputs
  consistency is possible via the `validate` method in the data class. This method accepts the
  configuration parameters of the associated processor if the latter are required for validation.

Warning: Type validation from the type hints is not automatically enforced by the data class
decorator.

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

Warning
Two possibilities to handle the input data in the subclass-specific `_pre_process` method:

- Extract the relevant input data to manipulate from the `input_data` dictionary (e.g. using
  the `get` method for dictionaries).
- Specify the relevant inputs in the signature of the `_pre_process` method to manipulate
  them directly within the method body. In this case, the method should still include a
  catch-all `**input_data` argument to be consistent with the signature of the ase class
  (Liskov Substitution Principle). Then, the dictionary `input_data` recovered within the
  method body contains the *remaining* inputs which are not isolated in the signature.
  Therefore, the method should not return the `input_data` dictionary, but rather
  reconstruct or update it to introduce the isolated inputs.

Conclusion: Keep the `**input_data` syntax without detailing. For consistency, follow this approach
also in the `_process` method and `_post_process` method (with the tuple `*output_data`). However,
inputs can be recovered differently in those methods:

- In `_pre_process`, extract the relevant inputs from the dictionary with the `get` method. This
  allows to specify a default value if the input is not found (like a default factory). If only
  input validation is performed without any modification (error raising), then only return the
  `input_data` dictionary. Otherwise, update the dictionary with the modified inputs before
  returning it.
- In `_process`, directly access the inputs from the dictionary using the bracket notation. The
  inputs are guaranteed to be present and valid since they have been validated in the `_pre_process`
  method. This approach avoids warning messages from the type checkers regarding the value that is
  accessed (contrary to the `get` method, since it should be necessary to specify again a default
  value matching the expected type for the subsequent processing steps).
- In the `_post_process` method, unpack the outputs from the tuple to access them directly within
  the method body by their index.

**Manipulating Inputs and Outputs through the Pipeline**

- Consistent formats: Within the `process` method, inputs are manipulated as a dictionary, while the
  outputs are handled as a tuple. Those formats are maintained across the `_pre_process`, `_process`
  and `_post_process` methods to ensure consistency throughout the pipeline. Thereby, modifications
  in any step does not affect the overall interface of the processor.
- Flexible signatures: In signature of the base `process` method supports any number of inputs
  passed as keyword arguments. Concrete processors's methods can specialize their signatures to
  specify the exact inputs require for their tasks. To ensure compatibility with the base class
  (Liskov Substitution Principle), the subclass signature should set default values to the named
  arguments from the input dataclass.
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


For each subclass, inputs are outputs are specified in the documentation of the
subclass-specific main `_process` method.

For input validation, pass the dictionary `input_data` received by the base processor's `process`
method to the subclass-specific `_pre_process` method. Unpack the keyword arguments within the
pre-processing method.

For output validation, pass the tuple `output_data` returned by the subclass-specific `_process`
method to the subclass-specific `_post_process` method. Unpack the return values within the
post-processing method. If the processor returns a single value, format it as a tuple.

Randomness
^^^^^^^^^^
Random state initialization is fully handled by the base class. Subclasses do not need to
manually define nor set any seed. A seed can be passed directly to the base `process` method as
an extra input.
"""
