"""
:mod:`utils.io` [subpackage]

Classes to handle input/output files manipulations.

Sub-Packages
-----------
:mod:`formats`
:mod:`savers`
:mod:`loaders`

Notes
-----
Goals of this Subpackage :

- Single Responsibility Principle
  File handling operations are centralized instead of beingscattered across the package. Data
  classes are not polluted data by input-output operations which would obscure their primary
  responsibilities.
- Consistency
  File handling operations are performed uniformly across the package via a common interface and
  utility methods.
- Flexibility and Scalability
  Modifications of the file handling rules do not require updates in data classes.
  New methods can be added without affecting the existing dataset classes.


Implementation
--------------
**Abstract Base Classes and Concrete Implementations**

PathRuler/Saver/Loader are abstract base classes, which define the interfaces used uniformly across
the whole package to generate paths and save/load data.
Each subclass of PathRuler implements the version for one specific data set (:meth:`get_path`).
Each subclass of Saver/Loader implements the version for one specific file format.

**Strategy Design Pattern and Composition**

Each object which needs to use those functionalities can select the appropriate subclass of
PathRuler/Saver/Loader which implements the desired version. It stores an instance of it and
delegate to it the responsibility of path generation.
This design is more extensible than a single class with multiple methods and a large conditional
statement to select one method via an argument.

**Arguments and Decoupling**

Any object using a PathRuler/Saver/Loader has to provide arguments taken from their attributes.
PathRuler/Saver/Loader do not accept custom objects as a whole from which they would extract the
necessary information. Thereby, it remains agnostic about the internal details of the classes which
use them.

See Also
--------
:mod:`pathlib`: Object-oriented filesystem paths (native library since Python 3.4).
    Alternative to :mod:`os.path` which represents paths as strings. Here, paths are represented as
    objects with methods and properties.
    Paths objects are compatible with common libraries which use paths to read/write files, without
    requiring conversion to strings (e.g. :func:`open`, :func:`np.savetxt`, :func:`pd.to_csv`)
    The library `pathlib` also handles differences between operating systems (POSIX, WindowsPath).
"""
