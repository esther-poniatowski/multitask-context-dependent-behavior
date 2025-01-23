#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`run_pipeline` [script]

Run a pipeline from the command line.

Functions
---------
main

Usage
-----
```
python run_pipeline.py pipeline --params path/to/params.yml --paths path/to/paths.yml
```

Arguments
---------
pipeline : str
    Name of the pipeline to execute. This name should match the name of the Python file in the
    `pipelines` directory.
--params : str
    Path to the configuration file storing parameters used in computations.
--paths : str
    Path to the configuration file specifying paths for input and output data.

Raises
------
FileNotFoundError
    If the configuration files are not found.
Exception
    If an error occurs while loading the configuration files.
ModuleNotFoundError
    If the pipeline is not found in the `pipelines` directory.
Exception
    If an error occurs while running the pipeline.

Notes
-----
Logging configuration:
- Log level: INFO
- Log format: timestamp, log level, and message
- Error messages are logged for all caught exceptions
"""

import argparse
import importlib
import logging

from utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODULE_PREFIX = "core.pipelines."
"""Path to the `pipelines` directory from the root of the project."""


def main():
    """
    Run a pipeline from the command line.

    See Also
    --------
    utils.load_config : Utility function to load configuration files in YAML format.
    argparse.ArgumentParser : Command-line argument parser.
    importlib.import_module : Import a module by name.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run a pipeline with given parameters and paths."
    )
    parser.add_argument(
        "pipeline",
        type=str,
        help="Name of the pipeline to execute"
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to the configuration file storing parameters used in computations"
    )
    parser.add_argument(
        "--paths",
        type=str,
        required=True,
        help="Path to the configuration file specifying paths for input and output data"
    )
    args = parser.parse_args()

    # Load configuration files
    try:
        params = load_config(args.params)
        paths = load_config(args.paths)
    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Failed loading configuration files: %s", e)
        raise

    # Dynamically import the pipeline
    module_name = f"{MODULE_PREFIX}.{args.pipeline}"
    try:
        pipeline_module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.error("Pipeline %s not found at %s", args.pipeline, module_name)
        raise

    # Run the pipeline
    try:
        pipeline_module.run(params, paths)
    except Exception as e:
        logger.error("Failed running pipeline: %s", e)
        raise

if __name__ == "__main__":
    main()
