#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.entities.exp_conditions` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
`ExpCondition`
"""
from types import MappingProxyType

from typing import Self, Iterable, List, Dict, Type, Set, FrozenSet, Mapping
from itertools import product
import warnings

import numpy as np

from core.entities.exp_factors import ExpFactor
from core.coordinates.exp_factor_coord import CoordExpFactor


class ExpCondition:
    """
    Experimental condition defined by the combination of several experimental factors, among:

    - task
    - attentional state
    - category of the stimulus
    - behavioral choice
    - response outcome

    Attributes
    ----------
    factors : List[str]
        Register of the attributes names for all specified factors.
    factor_types : Dict[Type[ExpFactor]], str]
        Mapping from the factor types (classes) and their attribute names.

    Methods
    -------
    `combine_factors`
    `__iter__`
    `__eq__`
    `__hash__`
    `__add__`
    `match`
    `count`

    Notes
    -----
    Partial conditions can be defined by specifying only the factors of interest. The remaining
    factors are set to `None` by default.

    Warning
    -------
    Instantiate with keyword arguments to avoid confusion between the factors, especially when
    partial conditions are defined.

    Examples
    --------
    Initialize a partial condition with three factors (without specifying the 'attention' factor):

    >>> task = Task('PTD')
    >>> category = Category('R')
    >>> behavior = Behavior('Go')
    >>> exp_cond_1 = ExpCondition(task=task, category=category, behavior=behavior)
    >>> exp_cond_1
    ExpCondition(task=PTD, category=R, behavior=Go)

    Initialize directly with strings:

    >>> exp_cond_2 = ExpCondition(task='CLK', attention='p', category='T')

    Retrieve the task:

    >>> exp_cond_1.task
    Task('PTD')

    See Also
    --------
    `ExpFactor`
    `Task`
    `Attention`
    `Category`
    `Behavior`
    `ResponseOutcome`
    `ExpConditionUnion`
    """

    def __init__(self, **factors: Dict[str, ExpFactor]) -> None:
        self.factors: List[str] = []
        self.factor_types: Dict[Type[ExpFactor], str] = {}
        for name, factor in factors.items():
            if not isinstance(factor, ExpFactor):
                raise TypeError(f"Invalid argument for ExpCondition: {name} not ExpFactor")
            setattr(self, name, factor)
            self.factors.append(name)
            self.factor_types[factor.__class__] = name

    def __iter__(self):
        """
        Iterate over the factors of the experimental condition.

        Yields
        ------
        name : str
            Name of the factor.
        fact : ExpFactor
            Factor instance.
        """
        for name in self.factors:
            yield name, getattr(self, name)

    def __repr__(self) -> str:
        return f"ExpCondition({', '.join(f'{name}={fact}' for name, fact in self)})"  # use __iter__

    def get_factor_name(self, factor_type: Type[ExpFactor]) -> str | None:
        """
        Get the attribute name corresponding to an experimental factor class.

        Arguments
        ---------
        factor_type : Type[ExpFactor]
            Class of the experimental factor for which to retrieve the attribute name.

        Returns
        -------
        name : str | None
            Name of the attribute corresponding to the factor type.
            None if the factor type is not present in the experimental condition instance.
        """
        return self.factor_types.get(factor_type, None)

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

        >>> exp_conds = ExpCondition.generate_conditions(
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

    def __eq__(self, other) -> bool:
        """
        Checks the equality between two experimental condition instances.

        Returns
        -------
        bool
            True if all the factors contained in both attributes are equal.

        See Also
        --------
        `Entity.__eq__`
        """
        # Check type ExpCond
        if not isinstance(other, self.__class__):
            return False
        # Check similar names
        names_1, names_2 = set(self.factors), set(other.factors)
        if names_1 != names_2:
            return False
        # Check factors equality
        for name in names_1:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self) -> int:
        """
        Hash the experimental condition instance based on the values of its factors.

        Returns
        -------
        int
            Hash value of the experimental condition instance.
        """
        return hash(tuple(getattr(self, name) for name in self.factors))

    def __add__(self, other: Self) -> "ExpConditionUnion":
        """
        Combine two conditions into a union of conditions.

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task='PTD', attention='a', category='R')
        >>> exp_cond_2 = ExpCondition(task='CLK', attention='p', category='T')
        >>> union = exp_cond_1 + exp_cond_2
        >>> union.conditions
        [ExpCondition(task='PTD', attention='a', category='R'),
         ExpCondition(task='CLK', attention='p', category='T')]
        """
        return ExpConditionUnion(self, other)

    def match(self, *coords: CoordExpFactor) -> np.ndarray:
        """
        Generate a boolean mask to index a set of samples based on a condition.

        Arguments
        ---------
        coords : CoordExpFactor
            Coordinates of experimental factors for each sample.

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match the condition.
            .. _mask:

        Raises
        ------
        UserWarning
            If the coordinate class is not handled for the types of factors stored in the
            experimental condition instance.
            If the type of factor associated to a coordinate is duplicated among the coordinates.
        ValueError
            If the number of samples in the provided coordinates is inconsistent.

        Examples
        --------
        >>> exp_cond = ExpCondition(task='PTD', attention='a', category='R')
        >>> task_coord = np.array(['PTD', 'CLK', 'PTD'])
        >>> attn_coord = np.array(['a', 'p', 'a'])
        >>> categ_coord = np.array(['R', 'T', 'R'])
        >>> mask = exp_cond.match(task=task_coord, attn=attn_coord, categ=categ_coord)
        >>> mask
        array([ True, False,  True])

        See Also
        --------
        `Coordinate.get_entity`
            Get the type of entity associated with the coordinate, which is expected to be an
            `ExpFactor` entity for `CoordExpFactor` instances.
        """
        # Filter coordinates based on the condition factors
        # Ignore coordinate if:
        # - it types is not handled by the condition
        # - its type has already been encountered in a previous coordinate (duplicate type)
        ignored: List[int] = []
        seen_types: Set[Type[ExpFactor]] = set()
        for i, coord in enumerate(coords):
            entity = coord.get_entity()  # retrieve the entity associated with the coordinate
            if entity not in self.factor_types or entity in seen_types:
                ignored.append(i)
        for i in ignored:
            warnings.warn(f"Ignored coordinates: {coords[i].__class__}", UserWarning)
        valid_idx = [i for i in range(len(coords)) if i not in ignored]
        # Initialize the mask with all True values
        n_samples = [len(coord) for coord in coords if coord not in ignored]
        if len(set(n_samples)) > 1:
            raise ValueError(f"Inconsistent number of samples across coordinates: {n_samples}")
        mask = np.ones(n_samples[0], dtype=bool)
        # Apply matching conditions for each coordinate only for specified instance factors
        for coord in [coords[i] for i in valid_idx]:
            name = self.get_factor_name(entity)  # attribute name for factor type
            if name is not None:  # always verified here, checked to silence type checker errors
                match_value = getattr(self, name)
                if match_value is not None:  # idem
                    mask &= coord == match_value
        return mask

    def count(self, *coords: CoordExpFactor) -> int:
        """
        Count the number of samples that match the condition.

        Arguments
        ---------
        coords : CoordExpFactor

        Returns
        -------
        n_samples : int
            Number of samples matching the condition.
        """
        mask = self.match(*coords)
        return np.sum(mask)


class ExpConditionUnion:
    """
    Union of experimental conditions.

    Attributes
    ----------
    exp_conds :
        Arbitrary number of condition objects.

    Methods
    -------
    `get`
    `__iter__`
    `match`
    `count`

    Examples
    --------
    >>> exp_cond_1 = ExpCondition(task='PTD', attention='a', category='R')
    >>> exp_cond_2 = ExpCondition(task='CLK', attention='p', category='T')
    >>> exp_cond_3 = ExpCondition(task='PTD', category='R', behavior='Go')
    >>> union = ExpConditionUnion(exp_cond_1, exp_cond_2, exp_cond_3)
    >>> union.get()
    [ExpCondition(task='PTD', attention='a', category='R'),
     ExpCondition(task='CLK', attention='p', category='T'),
     ExpCondition(task='PTD', category='R', behavior='Go')]
    """

    def __init__(self, *exp_conds: ExpCondition) -> None:
        self.exp_conds = list(exp_conds)

    def get(self) -> List[ExpCondition]:
        """Get the list of conditions in the union."""
        return self.exp_conds

    def __iter__(self):
        """Iterate over the conditions in the union."""
        return iter(self.exp_conds)

    def match(self, **coords: CoordExpFactor) -> np.ndarray:
        """
        Generate a boolean mask to index samples that match any of the conditions.

        Arguments
        ---------
        coords : CoordExpFactor

        Returns
        -------
        mask : np.ndarray
            Boolean mask to index samples that match any of the conditions.

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task='PTD', attention='a', category='R')
        >>> exp_cond_2 = ExpCondition(task='CLK', attention='a', category='T')
        >>> union = exp_cond_1 + exp_cond_2
        >>> task_coord = np.array(['PTD', 'CLK', 'PTD'])
        >>> attn_coord = np.array(['a', 'p', 'a'])
        >>> categ_coord = np.array(['R', 'T', 'R'])
        >>> mask = union.match(task=task_coord, attn=attn_coord, categ=categ_coord)
        >>> mask
        array([ True, False,  True])
        """
        # Get the masks for each condition in the union using the match method of ExpCondition
        all_masks = [cond.match(**coords) for cond in self.exp_conds]
        # Combine all masks with a logical OR operation
        mask = np.logical_or.reduce(all_masks)
        return mask

    def count(self, **coords: CoordExpFactor) -> int:
        """
        Count the number of samples that match any of the conditions.

        Arguments
        ---------
        coords : CoordExpFactor

        Returns
        -------
        n_samples : int
            Number of samples matching any of the conditions.
        """
        mask = self.match(**coords)
        return np.sum(mask)


class PipelineCondition(ExpCondition):
    """
    Base class for experimental conditions tailored to specific pipelines.

    Attributes
    ----------
    REQUIRED_FACTORS : Mapping[str, FrozenSet[ExpFactor]], default={}
        Experimental factors required by the pipeline.
        Keys: Names of the factors to use as attributes for the experimental conditions.
        Values: Allowed ExpFactor instances for each factor, whose common type corresponds to the
        type of the attribute in `ExpCondition.factor_types`.

    Methods
    -------
    generate:
        Generate all conditions for the pipeline based on the specified factors.

    Raises
    ------
    ValueError
        If the factors provided do not match the required factors for the pipeline.
        If some required factors are missing.
    """

    REQUIRED_FACTORS: Mapping[str, FrozenSet[ExpFactor]] = MappingProxyType({})

    def __init__(self, **factors: Dict[str, ExpFactor]) -> None:
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
