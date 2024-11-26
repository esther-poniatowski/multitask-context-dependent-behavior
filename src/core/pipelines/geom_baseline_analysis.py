#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.pipelines.geom_baseline_analysis` [module]

Classes
-------
`GeomBaselineAnalysis`

Notes
-----


"""
from core.pipelines.base_pipeline import Pipeline
from core.steps.concrete_steps import SelectUnits
from core.attributes.brain_info import Area
from core.composites.exp_conditions import ExpCondition
from core.attributes.exp_factors import Task, Attention, Category, Behavior

from utils.io_data.loaders import LoaderCSVtoList


class GeomBaselineAnalysis(Pipeline):
    """
    Pipeline for analyzing the geometry of the baseline neuronal activity.

    Attributes
    ----------
    path_units : Path | str
        Path to the file containing the units in the population.
    path_excluded : Path | str
        Path to the file containing the units to exclude from the analysis.

    Notes
    -----
    """

    REQUIRED_PATHS = frozenset(["path_units", "path_excluded"])

    def execute(self, **kwargs) -> None:
        """
        Implement the abstract method from the base class `Pipeline`.
        """
        # Identify PFC neurons
        all_units = LoaderCSVtoList(self.path_units).load()  # type: ignore[attr-defined] # pylint: disable=no-member
        excluded = LoaderCSVtoList(self.path_excluded).load()  # type: ignore[attr-defined] # pylint: disable=no-member
        selector = SelectUnits(all_units, excluded)
        units = selector.execute()

        # Build pseudo-trials
        # Define the conditions of interest
        tasks = [Task("PTD"), Task("CLK")]
        categories = [Category("R"), Category("T")]
        attentions = [Attention("a"), Attention("p")]
        behaviors = [Behavior("Go"), Behavior("NoGo")]
        exp_conds = ExpCondition.combine_factors(tasks, attentions, categories, behaviors)
