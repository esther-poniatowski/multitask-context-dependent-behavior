"""
mtcdb_etl
=========

Multi-Task-Context-Dependent-Behavior - Extraction - Transformation - Loading

Part of the MTCDB project dedicated to the ETL process.

Modules
-------
remote_access
    Manages remote access across multiple servers involved in the ETL process.
data_transfer
    Manages the transfer of raw data between different servers.
ingest
    Prepares data for processing.
"""
from importlib.metadata import version

__version__ = version(__package__)
