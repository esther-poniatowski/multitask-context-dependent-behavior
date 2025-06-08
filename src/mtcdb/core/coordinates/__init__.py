"""
`core.coordinates` [subpackage]

Classes representing coordinates (labels) associated to the data structures.

Each class of coordinate represents one common type of labels associated with the dimensions of the
data sets (e.g. time stamps, tasks, attentional state, stimuli...). Each is itself associated with
a class of the `Attribute` hierarchy, since its instances are arrays containing attribute values.

Each coordinate object encapsulates :

- Labels associated with one or several dimension(s) of a data set
- General metadata describing the constraints inherent to the coordinate family
- Specific metadata describing the unique properties of the coordinate instance
- Methods relevant to manipulate the coordinate labels

Modules
-------
`base_coordinate`
`time_coord`
`exp_factor_coord`
`exp_structure_coord`
`trials_coord`
`brain_info_coord`

Implementation
--------------
All coordinates adhere to a *uniform* interface established in the abstract base class `Coordinate`.

Each subtype of coordinate must implement the following steps:

- Define the `ATTRIBUTE` class attribute to specify the associated attribute class. Exception: Time
  coordinates.
- Define the `DTYPE` class attribute to specify the data type of the coordinate labels.
- Define the `METADATA` class attribute to specify the names of the additional attributes storing
  metadata alongside with the coordinate values (if any).
- Define the `SENTINEL` class attribute to specify the sentinel value marking missing or unset
  values in the coordinate labels.

Notes
-----
Coordinates do not store attributes which would depend on the data structure in which they are
embedded (e.g. name of the associated dimension in the data structure instance). This pairing is
managed by the data structures themselves. This design enhances modularity and decouples the
coordinate classes from the data set classes to which they apply.

Examples
--------

Create a basic coordinate from custom labels and metadata:

>>> values = np.array([True, False, True])
>>> coord = CoordError(values)

Create an 'empty' coordinate from a shape, filled with a sentinel value (here, -1):

>>> shape = (2, 3)
>>> coord = CoordFolds.from_shape(shape)
>>> print(coord)
[[ -1  -1  -1]
 [ -1  -1  -1]]

Generate basic labels with minimal arguments:

>>> CoordTime.build_labels(n_smpl=10, t_max=1)
<CoordTime> : 10 time points at 0.1 s bin.

"""
