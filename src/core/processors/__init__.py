#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.processors` [subpackage]

Classes implementing mid-level data processors used for data transformations in the analysis.

**Content and Functionalities of Processors**

Processors are designed to process data using a consistent framework across different subclasses.

- Each *class* represents one family of processors, corresponding to a specific transformation.
  Examples: data filtering, model fitting, signal analysis.
- Each *instance* represents one specific processor with a set of configuration parameters.
  Examples: filtering spikes with a specific threshold.

Each processor object encapsulates:

- Configuration parameters
- Input and output dynamic attributes for one specific data transformation.
- Methods for validating and processing the data.

**Processing Workflow**

The core of the processing workflow is managed by the main method `process`, which is called to
execute the processing on specific inputs.

Processors are created and used as follows:

+-----------+--------------------------------------+--------------------------------------+
| Step      | Set a Processor Instance             | Process Actual Data                  |
+===========+======================================+======================================+
| Approach  | ``ProcessorSubclass(config=...)``    | ``processor.process(**input_data)``  |
+-----------+--------------------------------------+--------------------------------------+
| Use Cases | - Instantiate with config settings   | - Apply processing logic to input    |
|           | - Define the behavior of the         |   data, gat target results           |
|           |   processor                          |                                      |
+-----------+--------------------------------------+--------------------------------------+

Modules
-------
:mod:`core.processors.base`

Implementation
--------------
Each specific processor is implemented as a concrete subclass of the base class :class:`Processor`.

Each concrete subclass define its own processing logic in two methods:

- `_validate_data` (optional), which is called before processing to validate the input data.
-  `_process` (required), which performs the actual processing steps.

Both those methods can call other internal helper methods corresponding to different processing
steps.

Examples
--------
Define a processor subclass which performs a basic data transformation:

.. code-block:: python

    class ExampleProcessor(Processor):
        input_attrs = ("input_arr",)
        output_attrs = ("output_arr",)
        proc_data_type = {
            "input_arr": np.ndarray,
            "output_arr": np.ndarray,
        }

        def _validate_data(self, **input_data):
            if not input_data["input_arr"].ndim == 1:
                raise ValueError(f"Invalid dimension for input data: {input_data['input_arr'].ndim}")

        def _process(self, **input_data):
            result = input_data["input_arr"] * 2
            return {"output_arr": result}

Use the processor on actual input data:

.. code-block:: python

    processor = ExampleProcessor()
    processor.process(input_data=np.array([1, 2, 3]))
    # Output: {'output_data': array([2, 4, 6])}
"""
