#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:mod:`core.builders` [subpackage]

Classes implementing builders for data structures used in the analysis.

Functionalities of each builder class:

- Collect components (e.g., features, metadata) to progressively build the associated data structure.
- Finalize the creation of the data structure (`build()` method) which may validate and handle any missing components.

Flexible configurations of data structured can be obtained by adding different components to the builder.

Notes
-----
Creating Data structures with builders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Instantiate the builder for the desired data structure
2. Add components to the builder (e.g., data, metadata)
3. Build the final object with the `build()` method

.. code-block:: python

      builder = DataBuilder()
      builder.add_data(data)
      builder.add_metadata(metadata)
      data_structure = builder.build()

Builders and Pipelines Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pipelines are used to produce components which are incorporated in the final data structure together
with other relevant metadata or attributes.


Modules
-------


See Also
--------
:mod:`core.data_structures`
    Data structure classes which are constructed by the builders.
:mod:`core.processors`
    Pipelines used to process and transform data before final assembly by the builders.
"""
