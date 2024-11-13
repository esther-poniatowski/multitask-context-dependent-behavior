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
from pathlib import Path
from typing import Optional, Type, FrozenSet

from core.steps.base_step import Step


class Pipeline(ABC):
    """
    Base class for analysis pipelines.

    This class defines the interface for orchestrating a complete analysis workflow, including data
    loading, processing, and saving results.

    Class Attributes
    ----------------
    REQUIRED_PATHS : FrozenSet[str]
        Set of required paths for the pipeline execution.

    Attributes
    ----------
    ready : bool
        Flag indicating whether the pipeline is ready to execute, i.e. all required paths are set.

    Methods
    -------
    set_path
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

    REQUIRED_PATHS: FrozenSet[str] = frozenset()

    def __init__(self) -> None:
        """Initialize the pipeline state and the required paths to None."""
        self.ready = False
        for path in self.REQUIRED_PATHS:
            setattr(self, path, None)

    def set_path(self, attr, path) -> None:
        """
        Set the path to one input or output file required for the pipeline execution.

        Arguments
        ---------
        attr : str
                Name of the path attribute to set.
        path : Path | str
                Actual path to the file for input or output.
        """
        assert attr in self.REQUIRED_PATHS, f"Invalid path attribute: {attr}"
        if isinstance(path, str):
            path = Path(path)
        setattr(self, attr, path)

    @abstractmethod
    def execute(self, **kwargs) -> None:
        """
        Execute all the steps of the analysis.

        Arguments
        ---------
        kwargs
            Additional keyword arguments to pass to the pipeline steps.
        """
