"""
`core.attributes` [subpackage]

Class hierarchy representing key qualitative and quantitative "attributes" (i.e. descriptors,
properties) of the objects manipulated in the code base.

Modules
-------
`base_attribute`
`exp_factors`
`exp_condition`
`exp_structure`
`brain_info`
`trial_analysis_labels`

Notes
-----
The hierarchy provides a consistent interface for handling diverse data types while allowing for
custom behavior in each subclass. Each subclass of `Attribute` corresponds to a specific type
of descriptor, which represents one aspects of the experiment or the analysis. They include:

- Neuronal information (qualitative properties): brain area, cortical depth, animal...
- Conditions of the experimental paradigm (qualitative factors): task, attentional state,
  stimulus...
- Structure of the experiment (quantitative factors): recording number, position of the stimulus in
  a sequence...
- Trial-related labels for analysis: folds, indices of pseudo-trials...

Most of the coordinates aim to describe trial-level properties, except those which describe
neuronal information.

The base class `Attribute` provides core functionality, shared by many subclasses:

- Interacting with allowed values (`OPTIONS`).
- Validating input values against the allowed set.

Each subclass of `Attribute` inherits from a basic type (e.g., `int`, `str`) and extends it with is
own constraints and domain-specific logic.

- Qualitative attributes mimic "Enum" classes, with a fixed set of allowed values.
- Quantitative attributes are more flexible, some may impose range constraints or a single boundary.


Examples
--------
Get the valid options for Attention objects :

>>> from core.attributes import Attention
>>> print(list(Attention.get_options()))
('a', 'p', 'p-pre', 'p-post')

Get the attentional states for naive animals :

>>> print(Attention.naive())
('p')

Get the full label for the attentional state 'a' :

>>> attn = Attention('a')
>>> print(attn.full_label)
'Active'

"""
