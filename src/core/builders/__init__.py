"""
`core.builders` [subpackage]

Classes implementing builders (creators) for **data structures**.

Functionalities of each builder class:

- Collect components (e.g., features, metadata) to progressively build the associated data
  structure.
- Finalize the creation of the data structure (`build()` method) which may validate and handle any
  missing components.

Flexible configurations of data structured can be obtained by adding different components to the
builder.

Modules
-------
`base_builder`

Notes
-----
Creating Data structures with builders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Instantiate the builder for the desired data structure
2. Add components to the builder (e.g., data, metadata)
3. Build the final object with the `build()` method

.. code-block:: python

      builder = Builder[DataStructure]()
      builder.add_data(data)
      builder.add_metadata(metadata)
      data_structure = builder.build()

Builders and Pipelines Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pipelines are used to produce components which are incorporated in the final data structure together
with other relevant metadata or attributes.


Comparison: Builders vs. Processors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Responsibilities:

- Processor: Perform a focused (often complex) computation, transformation or algorithm on input
  data.
- Builder: Produce a data structure. It involves orchestrating multiple steps: invoking processors,
  handling data flow between them, combining their results, assembling components.

Organization of the Operations:

- Processor: Clear structure, adapted for the template method pattern. Method names reflect a strict
  workflow (pre-processing, processing, post-processing).
- Builder: Flexible structure, tailored to the assembly process. Method names reflect the sequence
  of steps involved in the assembly.

Input/Output Handling:

- Processor: Inputs are strictly validated, and outputs are returned as tuples or single values.
- Builder: Input validation is less rigid, it may accept a wider variety of inputs. It returns a
  single fully constructed object.

Statefulness:

- Processor: Processors are stateless each processing call is independent of the others.
- Builder: Builders are often *stateful*. They may need to store intermediate results between steps
  or manage temporary states during the assembly process.

Examples
--------
Implementation of a concrete builder class:

.. code-block:: python

    class ConcreteBuilder(Builder[DataStructure]):
        data_class = ConcreteDataStructure

        def build(self,
                  input_for_data: np.ndarray,
                  input_for_coord: np.ndarray,
                  input_metadata: str
        ) -> ConcreteDataStructure:
            data_pipeline = DataPipeline()
            coord_pipeline = CoordPipeline()
            data = data_pipeline.process(input_for_data)
            coord = coord_pipeline.process(input_for_coord)
            return self.data_class(data=data, coord=coord, metadata=input_metadata)

Usage of the concrete builder:

.. code-block:: python

        builder = ConcreteBuilder()
        data_structure = builder.build(input_for_data, input_for_coord, input_metadata)

See Also
--------
`core.data_structures`
    Data structure classes which are constructed by the builders.
`core.processors`
    Pipelines used to process and transform data before final assembly by the builders.

"""
