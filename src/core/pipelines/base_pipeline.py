#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.pipelines.base_pipeline` [module]

Classes
-------
`Pipeline`

Notes
-----
Each subclass of `Pipeline` should  inherits from this class.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Any


@dataclass
class PipelineConfig(ABC):
    """
    Abstract base class for pipeline configurations.

    Define the arguments for the pipeline constructor, which fix its behavior for a number of runs.
    """


@dataclass
class PipelineInputs(ABC):
    """
    Abstract base class for runtime inputs of a pipeline execution.

    Define the arguments of the `execute` method of the pipeline, for a single run.
    """


C = TypeVar("C", bound=PipelineConfig)
I = TypeVar("I", bound=PipelineInputs)


class Pipeline(ABC, Generic[C, I]):
    """
    Abstract base class for analysis pipelines.

    This class defines the interface for orchestrating a complete analysis workflow, including data
    loading, processing, and saving results.

    Class Attributes
    ----------------

    Attributes
    ----------

    Methods
    -------
    execute

    See Also
    --------
    `core.steps.base_step.Step`

    Notes
    -----
    Contrary to the ordinary Pipeline design pattern, this class is not aimed to flexibly add steps
    and chain them together. Instead, it is designed to execute fixed sequences of steps
    implemented in each specific subclass.

    Implementation
    --------------
    TODO: Implement the following functionalities:

    Checkpointing
    ^^^^^^^^^^^^^
    The `load_checkpoint` and `save_checkpoint` methods of each step are called before and after
    its execution to save the state of the pipeline in case of failure.

    Error Handling
    ^^^^^^^^^^^^^^
    If an error occurs during the execution of a step, the pipeline stops and prints the error
    message. This allows to identify the step that caused the error and start the pipeline from
    this steps in the next run.
    """

    def __init__(self, config: C, **kwargs: Any) -> None:
        """
        Initialize the pipeline state.

        Arguments
        ---------
        config : PipelineConfig
            Configuration for the pipeline.
        kwargs
            Additional keyword arguments to pass to the pipeline.
        """
        self.config = config

    @abstractmethod
    def execute(self, inputs: I, **kwargs: Any) -> None:
        """
        Execute all the steps of the analysis.

        Arguments
        ---------
        inputs : PipelineInputs
            Input data for the pipeline.
        kwargs
            Additional keyword arguments to pass to the execution steps.
        """
