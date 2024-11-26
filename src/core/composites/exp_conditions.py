#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.exp_conditions` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
ExpCondition
PipelineCondition
"""
from types import MappingProxyType

from typing import Self, Iterable, List, Dict, Type, FrozenSet, Mapping
from itertools import product

from core.attributes.exp_factors import ExpFactor
from core.composites.strata import Stratum, StrataUnion


class ExpCondition(Stratum):
    """
    Experimental condition defined by the combination of several experimental factors.

    Usual experimental factors include:

    - task
    - attentional state
    - category of the stimulus
    - behavioral choice
    - response outcome

    Methods
    -------
    combine_factors

    Notes
    -----
    All the methods from the `Stratum` class are available for the `ExpCondition` class. The new
    behavior of this class is to restrict the factors to the *experimental* factors only.

    See Also
    --------
    Stratum
    """

    # --- Create ExpCondition instances ------------------------------------------------------------

    def __init__(self, **factors: ExpFactor) -> None:
        for name, factor in factors.items():
            if not isinstance(factor, ExpFactor):
                raise TypeError(f"Invalid argument for ExpCondition: {name} not ExpFactor")
        super().__init__(**factors)

    @staticmethod
    def combine_factors(*selected_factors: Iterable[ExpFactor] | ExpFactor) -> List["ExpCondition"]:
        """
        Generate experimental conditions by combining a set of experimental factors.

        Arguments
        ---------
        selected_factors : Iterable[ExpFactor] | ExpFactor
            Instances of the experimental factors to consider to generate the experimental
            conditions of interest. Each factor must match a recognized type in
            `ExpCondition.FACTOR_TO_ATTR`.

        Returns
        -------
        exp_conds : List[ExpCondition]
            Experimental conditions generated from the Cartesian product of the selected factor
            instances.

        Examples
        --------
        Generate all possible conditions for two tasks, a fixed attentional state, two stimuli and
        both behaviors:

        >>> exp_conds = ExpCondition.combine_factors(
        ...     tasks=[Task("PTD"), Task("CLK")],
        ...     attentions=Attention("a"),
        ...     categories=[Category("R"), Category("T")],
        ... )
        >>> exp_conds
        [ExpCondition(task=PTD, attention=a, category=R),
         ExpCondition(task=PTD, attention=a, category=T),
         ExpCondition(task=CLK, attention=a, category=R),
         ExpCondition(task=CLK, attention=a, category=T)]

        Notes
        -----
        For each factor, if a single instance is provided, it considered as fixed and will be used
        for all the conditions.
        If no instance is provided, it is set to `None` in all the experimental conditions.
        To retrieve all the allowed values for experimental factor, use the `get_factors` method of
        the factor class.

        See Also
        --------
        `itertools.product`: Cartesian product of input iterables.
        """
        # Format to list for consistency
        sequences = [[fact] if isinstance(fact, ExpFactor) else fact for fact in selected_factors]
        # Generate all combinations of the provided factors
        combinations = product(*sequences)  # List[Tuple[ExpFactor, ...]]
        # Create and return a list of ExpCondition instances from the combinations
        return [ExpCondition(*comb) for comb in combinations]

    def __repr__(self) -> str:
        return f"ExpCondition({', '.join(f'{name}={fact}' for name, fact in self)})"  # use __iter__


class ExpConditionUnion(StrataUnion):
    """
    Union of experimental conditions.
    """


class PipelineCondition(ExpCondition):
    """
    Base class for experimental conditions tailored to specific pipelines.

    Class Attributes
    ----------------
    REQUIRED_FACTORS : Mapping[str, FrozenSet[ExpFactor]], default={}
        Experimental factors required by the pipeline.
        Keys: Names of the factors to use as attributes for the experimental conditions.
        Values: Allowed ExpFactor instances for each factor, whose common type corresponds to the
        type of the attribute in `ExpCondition.factor_types`.

    Methods
    -------
    generate

    Raises
    ------
    ValueError
        If the factors provided do not match the required factors for the pipeline.
        If some required factors are missing.
    """

    REQUIRED_FACTORS: Mapping[str, FrozenSet[ExpFactor]] = MappingProxyType({})

    def __init__(self, **factors: ExpFactor) -> None:
        """Enforce the pipeline constraints."""
        valid_factors = {}
        for name, value in factors.items():
            if name in self.REQUIRED_FACTORS and value in self.REQUIRED_FACTORS[name]:
                valid_factors[name] = value
        missing = set(self.REQUIRED_FACTORS) - set(valid_factors)
        if missing:
            raise ValueError(f"Missing factors for {self.__class__.__name__}: {missing}")
        super().__init__(**valid_factors)

    @classmethod
    def generate(cls) -> ExpConditionUnion:
        """
        Generate all the valid experimental conditions based on the required factors.

        Returns
        -------
        exp_conditions : ExpConditionUnion
            All the valid experimental conditions for the pipeline.
        """
        selected_factors = list(cls.REQUIRED_FACTORS.values())
        valid_factors = cls.combine_factors(*selected_factors)  # list of ExpCondition
        return ExpConditionUnion(*valid_factors)
