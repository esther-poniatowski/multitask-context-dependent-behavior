"""
`core.processors.base_step` [module]

Classes
-------
`Step`

"""
from abc import ABC, abstractmethod
from typing import Any


class Step(ABC):
    """
    Base class for pipeline steps. Define the common interface to run a step.

    Methods
    -------
    execute
    load_checkpoint
    save_checkpoint

    Notes
    -----
    Each subclass of `Step` should inherit from this class.

    Checkpointing
    ^^^^^^^^^^^^^
    See the description in the base pipeline class `core.pipelines.base_pipeline.Pipeline`.

    Configuration
    ^^^^^^^^^^^^^
    See the description in the base pipeline class `core.pipelines.base_pipeline.Pipeline`.

    """

    @abstractmethod
    def execute(self) -> None:
        """Execute the step of the pipeline."""

    @abstractmethod
    def load_checkpoint(self) -> Any:
        """
        Load the checkpoint of the step.

        Returns
        -------
        Any
            Checkpoint data to input in the step.
        """

    @abstractmethod
    def save_checkpoint(self) -> None:
        """
        Save the checkpoint of the step.
        """
