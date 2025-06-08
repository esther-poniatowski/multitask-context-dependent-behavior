"""
`core.data_components.core_metadata` [module]

Classes
-------
MetaDataField
"""

from dataclasses import dataclass
from typing import Any, Type


@dataclass
class MetaDataField:
    """
    Metadata field for a data component.

    Attributes
    ----------
    field_name : str
        Name of the metadata field.
    field_type : Type[Any]
        Type of the metadata field.
    default_value : Any
        Default value for the metadata field.

    Notes
    -----
    This class can be used in both data components and data structures to specify the metadata
    fields or the identifiers.

    Examples
    --------
    Define a metadata field for a `CoreData` class:

    >>> class CoreTime(CoreData):
    ...     METADATA: Dict[str, MetadataField] = {
    ...         "origin": MetadataField(str, None),
    ...         "time_unit": MetadataField(str, "sec"),
    ...     }

    """

    field_type: Type[Any]
    default_value: Any
