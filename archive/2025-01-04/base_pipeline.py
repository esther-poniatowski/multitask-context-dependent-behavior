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
from typing import FrozenSet

from utils.io_data.base_loader import Loader
from utils.io_data.base_saver import Saver
from utils.storage_rulers.base_path_ruler import PathRuler


@dataclass
class PipelineConfig:
    """
    Configuration for the pipeline.

    Define the arguments of the constructor of the pipeline, which fix its behavior for a number of
    runs.
    """


@dataclass
class PipelineInputs:
    """
    Base class for the input data of an analysis pipeline.

    Define the arguments of the `execute` method of the pipeline, required for a single run.
    """


class Pipeline(ABC):
    """
    Base class for analysis pipelines.

    This class defines the interface for orchestrating a complete analysis workflow, including data
    loading, processing, and saving results.

    Class Attributes
    ----------------
    PATH_INPUTS : FrozenSet[PathRuler]
        Attribute names for the path rulers required to specify the paths to the input data.
    PATH_OUTPUTS : FrozenSet[PathRuler]
        Attribute names for the path rulers required to specify the paths to the output data.
    LOADERS : FrozenSet[Loader]
        Attribute names for the loaders required to load the input data.
    SAVERS : FrozenSet[Saver]
        Attribute names for the savers required to save the output data.

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

    PATH_INPUTS: FrozenSet[str] = frozenset()
    PATH_OUTPUTS: FrozenSet[str] = frozenset()
    LOADERS: FrozenSet[str] = frozenset()
    SAVERS: FrozenSet[str] = frozenset()

    def get_required_io_handlers(self) -> FrozenSet[str]:
        """
        Get the names of all the required IO handlers.

        Returns
        -------
        FrozenSet[str]
            Names of the required IO handlers.
        """
        return self.PATH_INPUTS | self.PATH_OUTPUTS | self.LOADERS | self.SAVERS

    def __init__(self) -> None:
        """Initialize the pipeline state and the required attributes to None."""
        self.ready = False
        for attr in self.get_required_io_handlers():
            setattr(self, attr, None)

    def set_io(self, attr, handler) -> None:
        """
        Set the handler to perform a required functionality, designated by the attribute.

        Arguments
        ---------
        attr : str
            Name of the attribute to set among the IO_HANDLERS.
        handler : type
            Class of the io handler selected to perform the required functionality.

        Raises
        ------
        AttributeError
            If the attribute is not valid for an IO handler.
        TypeError
            If the handler type is not valid.
        """
        if not attr in self.get_required_io_handlers():
            raise AttributeError(f"Invalid attribute: {attr}")
        if attr in self.PATH_INPUTS | self.PATH_OUTPUTS:
            if not issubclass(handler, PathRuler):
                raise TypeError(f"Invalid handler type: {handler} not a PathRuler")
        elif attr in self.LOADERS:
            if not issubclass(handler, Loader):
                raise TypeError(f"Invalid handler type: {handler} not a Loader")
        elif attr in self.SAVERS:
            if not issubclass(handler, Saver):
                raise TypeError(f"Invalid handler type: {handler} not a Saver")
        setattr(self, attr, handler)

    @abstractmethod
    def execute(self, **kwargs) -> None:
        """
        Execute all the steps of the analysis.

        Arguments
        ---------
        kwargs
            Additional keyword arguments to pass to the pipeline steps.
        """
