#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors` [subpackage]

Classes implementing mid-level data processors used for data transformations in the analysis.

Modules
-------
:mod:`core.processors.base_processor`

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


    class ExampleProcessorOutputs(ProcessorOutput):
        output_arr: np.ndarray = np.empty(0)


    class ExampleProcessor(Processor):

        # --- Define processor class-level attributes

        config_params = ("min_length",) # config parameters fixed for the processor instance
        input_dataclass = ExampleProcessorInputs
        output_dataclass = ExampleProcessorOutputs
        is_random = True

        # Call the parent constructor to set the config
        def __init__(self, min_length: int = 1):
            super().__init__(min_length=min_length)

        # --- Define processor methods

        def _preprocess(self, **input_data):
            # Subclass-specific validation
            if len(input_data["input_arr"]) < min_length:
                raise ValueError(f"Invalid length for input: {len(input_data['input_arr'])}")
            # Default validation: call the parent method
            super()._preprocess(**input_data)

        # Specify the signature for this concrete processor
        # Set default values to named arguments from the ExampleProcessorInput dataclass
        # to ensure compatibility with the base class (Liskov Substitution Principle)
        def _process(self, input_arr: np.ndarray = ExampleProcessorInputs.input_arr) -> np.ndarray:
            output_arr = self.input_arr * 2
            return output_arr

Use the processor on actual input data:

.. code-block:: python

    processor = ExampleProcessor(min_length=3)
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
- Automatic Validation: The structure of the inputs and outputs (including number, name, default
  values) is enforced via the data classes. This ensures a seamless flow through the processing
  pipeline when chaining several methods. In contrast, a dictionary-based approach using a
  class-level attribute defined in the processor would require manual validation and potentially
  clutter the processing logic.

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

"""
