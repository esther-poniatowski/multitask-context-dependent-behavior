"""
`run_pipeline` [script]

Run a pipeline from the command line using configuration files.

Functions
---------
main : Main entry point to run a pipeline from the command line.

Usage
-----
Run a pipeline with default configurations:

.. code-block:: sh

    python run_pipeline.py pipeline=my_pipeline

Override a single parameter using the dot notation:

.. code-block:: sh

    python run_pipeline.py pipeline=my_pipeline params.area=A1

Switch to a different configuration file:

.. code-block:: sh

    python run_pipeline.py pipeline=my_pipeline paths.data_dir=/path/to/data


Arguments
---------
pipeline : str
    Name of the pipeline to execute. This name should match the name of the Python file in the
    `pipelines` directory.
# TODO: Are the next two arguments necessary? How should I mention commonly overridden parameters?
--params : str
    Path to the configuration file storing parameters used in computations.
--paths : str
    Path to the configuration file specifying paths for input and output data.

Raises
------
# TODO: Shold I specify any exceptions raised by the functions called in this script? Or should I
document them in the `main` function?

Notes
-----
**Logging configuration**
The logging configuration is set in the `hydra/logging.yml` configuration file.

Default settings:

- Log level: INFO
- Log format: timestamp, log level, and message
- Error messages are logged for all caught exceptions

**Multi-level hierarchical configuration**

Configuration structure:

- config/: Directory containing configuration files
- main.yaml: Top-level configuration file (entry point for Hydra, specified in the `@hydra.main`
  decorator by the `config_path` arguments)
- pipeline/*.yaml: Pipeline-specific configurations
- params/*.yaml: Parameter configurations for computations
- paths/*.yaml: Input/output paths specifications for local data storage
- hydra/*.yaml: Hydra-specific configurations

Hierarchical merging:

Configurations are merged in the order of precedence specified in the `defaults` key of the main
configuration file.

Required arguments:

The `pipeline` argument is made mandatory via the main configuration file, with a dedicated syntax:
`pipeline: ???` or `pipeline: null`.

**Dynamic instantiation**

The pipeline class is instantiated dynamically using the `__target__` key in the configuration files
of each pipeline. It specifies the fully qualified name of the class to instantiate.

- Each pipeline is a class with a standardized `run()` method.
- Pipeline-specific parameters are dynamically passed to the instance.

Hydra Features Used:
- `__target__`: Dynamic class instantiation.

- `hydra.run.dir`: Controlling Hydra output storage.

See Also
--------
hydra : Configuration management tool for Python projects.
    Version: 1.3+ (forward-compatible)
"""

import logging
import hydra
from omegaconf import DictConfig

CONFIG_DIR = "config"
"""Name of the directory containing configuration files."""
CONFIG_MAIN = "main"
"""Name of the main configuration file."""


# Retrieve the logger automatically configured by Hydra
logger = logging.getLogger(__name__)

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_MAIN, version_base="1.3")
def main(cfg: DictConfig):
    """
    Run a pipeline with configurations managed by Hydra.

    Parameters
    ----------
    cfg : DictConfig
        Dictionary-like object containing merged configuration from Hydra.

    Raises
    ------
    ValueError
        If pipeline is not specified in the configuration.
    Exception
        If an error occurs while running the pipeline.

    See Also
    --------
    hydra.main : Decorator for the main entry point of a Hydra application.
    hydra.utils.instantiate : Instantiate an object from a configuration.
    """
    # Enforce the "pipeline" argument
    if "pipeline" not in cfg or cfg.pipeline is None:
        raise ValueError("Pipeline must be specified. Use `pipeline=pipeline_name`.")

    logger.info("Running pipeline: %s", cfg.pipeline._target_) # pylint: disable=protected-access
    logger.info("Configuration:\n%s", cfg.pretty())

    # Instantiate the pipeline dynamically and pass the configuration
    try:
        pipeline_instance = hydra.utils.instantiate(cfg.pipeline)
    except Exception as e:
        logger.error("Failed instantiating pipeline: %s", e)
        raise

    # Run the pipeline
    try:
        pipeline_instance.run()
    except Exception as e:
        logger.error("Failed running pipeline: %s", e)
        raise

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
