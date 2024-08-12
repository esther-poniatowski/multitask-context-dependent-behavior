"""
:mod:`mtcdb.preprocess.pipelines` [module]



See Also
--------

"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve
from typing import Literal, TypeAlias

from core.constants import T_BIN
from core.types import ArrayLike, NumpyArray


class Builder(ABC):
    @abstractmethod
    def build(self, data):
        pass

class BuilderA(Builder):
    # Focus on transformation logic without handling loading or saving directly.
    def build(self, raw_data: RawData):
        # Transformation logic for BuilderA
        transformed_attributes = ... # Transform raw_data.data
        return ProcessedData(transformed_attributes)

class BuilderB(Builder):
    def build(self, processed_data: ProcessedData):
        # Perform transformation logic from processed_data to final_attributes
        final_attributes = ...  # Transform processed_data.processed_attributes to final form
        return FinalData(final_attributes)


# -----

class Pipeline(ABC):
    @abstractmethod
    def add_step(self, builder: Builder):
        pass

    @abstractmethod
    def execute(self, data):
        pass

class DataPipeline(Pipeline):
    #  Ensures that both transformations
    # (from RawData to ProcessedData and from ProcessedData to FinalData)
    # are applied in sequence for each set of parameters.
    def __init__(self):
        self.steps = []

    def add_step(self, builder: Builder):
        self.steps.append(builder)

    def execute(self, raw_data: RawData):
        for session, unit in parameters:
            raw_data = RawData(session, unit)
            raw_data.load_data()

            data = raw_data
            for step in self.steps:
                data = step.build(data)

            data.save_data()


# Define parameters for sessions and units
parameters = [('session1', 'unit1'), ('session1', 'unit2'), ('session2', 'unit1')]

# Create a pipeline and add steps
pipeline = DataPipeline()
pipeline.add_step(BuilderA())
pipeline.add_step(BuilderB())

# Execute the pipeline on raw data and save the processed data
pipeline.execute(parameters)
