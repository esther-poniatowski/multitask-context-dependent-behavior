#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.composites.exp_conditions` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
ExpCondition
ExpConditionUnion
PipelineCondition
"""
from types import MappingProxyType

from typing import Self, Iterable, List, Dict, Type, FrozenSet, Mapping
from itertools import product

from core.entities.exp_factors import ExpFactor


class ExpCondition:
    """
    Experimental condition defined by the combination of several experimental factors.

    Usual experimental factors include:

    - task
    - attentional state
    - category of the stimulus
    - behavioral choice
    - response outcome

    Attributes
    ----------
    registry : List[str]
        Registry of the attributes names for all specified factors.
    factor_types : Dict[Type[ExpFactor]], str]
        Mapping from the factor types (classes) and their attribute names.

    Methods
    -------
    set_factor
    add_factor
    combine_factors
    get_factor
    get_entity
    __iter__
    __eq__
    __hash__
    __add__

    Notes
    -----
    Partial conditions can be defined by specifying only the factors of interest.

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

    Retrieve the task:

    >>> exp_cond_1.task
    Task('PTD')

    See Also
    --------
    ExpFactor
    ExpConditionUnion
    """

    # --- Create ExpCondition instances ------------------------------------------------------------

    def __init__(self, **factors: ExpFactor) -> None:
        self.registry: List[str] = []
        self.factor_types: Dict[Type[ExpFactor], str] = {}
        for name, factor in factors.items():
            self.set_factor(name, factor)

    def set_factor(self, name: str, factor: ExpFactor) -> None:
        """
        Set a new experimental factor to the condition after validation.

        Arguments
        ---------
        name : str
            Name of the attribute to use for the factor.
        factor : ExpFactor
            Experimental factor to add to the condition.
        """
        if not isinstance(factor, ExpFactor):
            raise TypeError(f"Invalid argument for ExpCondition: {name} not ExpFactor")
        setattr(self, name, factor)
        self.registry.append(name)
        self.factor_types[factor.__class__] = name

    def add_factor(self, name: str, factor: ExpFactor) -> "ExpCondition":
        """
        Add a new experimental factor to a condition.

        Arguments
        ---------
        factor : ExpFactor
            Experimental factor to add to the condition.

        Returns
        -------
        new_exp_cond : ExpCondition
            New experimental condition instance with the added factor.

        Examples
        --------
        >>> exp_cond = ExpCondition(task=Task('PTD'), attention=Attention('a'))
        >>> new_exp_cond = exp_cond.add_factor(Category('R'))
        >>> new_exp_cond
        ExpCondition(task=PTD, attention=a, category=R)
        """
        # Copy the current condition
        new_exp_cond = self.__class__(**{name: getattr(self, name) for name in self.registry})
        # Add the new factor
        new_exp_cond.set_factor(name, factor)
        return new_exp_cond

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

    # --- Get ExpCondition properties --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ExpCondition({', '.join(f'{name}={fact}' for name, fact in self)})"  # use __iter__

    def get_factor(self, factor_type: Type[ExpFactor]) -> ExpFactor | None:
        """
        Get the stored value corresponding to one experimental factor class.

        Arguments
        ---------
        factor_type : Type[ExpFactor]
            Class of the experimental factor for which to retrieve the value.

        Returns
        -------
        factor : ExpFactor
            Value of the factor if present in the experimental condition instance.
            None if the factor type is not present in the experimental condition instance.

        Examples
        --------
        >>> exp_cond = ExpCondition(task=Task('PTD'), attention=Attention('a'))
        >>> exp_cond.get_factor(Task)
        Task('PTD')
        """
        name = self.factor_types.get(factor_type, None)
        if name is not None:
            return getattr(self, name, None)
        return None

    def get_entity(self, name: str) -> Type[ExpFactor] | None:
        """
        Get the factor class corresponding to one experimental factor attribute.

        Arguments
        ---------
        name : str
            Name of the experimental factor attribute to retrieve.

        Returns
        -------
        factor_type : Type[ExpFactor]
            Class of the experimental factor corresponding to the attribute name.
            None if the attribute name is not present in the experimental condition instance.

        Examples
        --------
        >>> exp_cond = ExpCondition(task=Task('PTD'), attention=Attention('a'))
        >>> exp_cond.get_entity('task')
        Task
        """
        factor = getattr(self, name, None)
        factor_type = factor.__class__ if factor is not None else None
        return factor_type

    def get_entities(self) -> List[Type[ExpFactor]]:
        """
        Get all the factor classes present in the experimental condition instance.

        Returns
        -------
        factor_types : List[Type[ExpFactor]]
            List of the classes of the experimental factors present in the experimental condition.

        Examples
        --------
        >>> exp_cond = ExpCondition(task=Task('PTD'), attention=Attention('a'))
        >>> exp_cond.get_all_entities()
        [Task, Attention]
        """
        return [factor.__class__ for name, factor in self]

    # --- Magic methods ----------------------------------------------------------------------------

    def __iter__(self):
        """
        Iterate over the factors specified in the experimental condition.

        Yields
        ------
        name : str
            Name of the factor.
        fact : ExpFactor
            Factor instance.
        """
        for name in self.registry:
            yield name, getattr(self, name)

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

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task=Task('PTD'), attention=Attention('a'))
        >>> exp_cond_2 = ExpCondition(task=Task('CLK'), attention=Attention('a'))
        >>> exp_cond_1 == exp_cond_2
        False
        """
        # Check type ExpCond
        if not isinstance(other, self.__class__):
            return False
        # Check similar names
        names_1, names_2 = set(self.registry), set(other.registry)
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
        return hash(tuple(getattr(self, name) for name in self.registry))

    def __add__(self, other: Self) -> "ExpConditionUnion":
        """
        Combine two conditions into a union of conditions.

        Examples
        --------
        >>> exp_cond_1 = ExpCondition(task=Task('PTD'), attention=Attention('a'))
        >>> exp_cond_2 = ExpCondition(task=Task('CLK'), attention=Attention('p'))
        >>> union = exp_cond_1 + exp_cond_2
        >>> union.get()
        [ExpCondition(task='PTD', attention='a'),
         ExpCondition(task='CLK', attention='p')]
        """
        return ExpConditionUnion(self, other)


class ExpConditionUnion:
    """
    Union of experimental conditions, behaving like a list of conditions.

    Attributes
    ----------
    exp_conds :
        Arbitrary number of condition objects.

    Methods
    -------
    to_list
    __iter__

    Raises
    ------
    TypeError
        If any of the arguments is not an `ExpCondition` instance

    Examples
    --------
    >>> exp_cond_1 = ExpCondition(task=Task('PTD'), attention=Attention('a'))
    >>> exp_cond_2 = ExpCondition(task=Task('CLK'), attention=Attention('p'))
    >>> exp_cond_3 = ExpCondition(category=Category('R'), behavior=Behavior('Go'))
    >>> union = ExpConditionUnion(exp_cond_1, exp_cond_2, exp_cond_3)
    >>> union.to_list()
    [ExpCondition(task='PTD', attention='a'),
     ExpCondition(task='CLK', attention='p'),
     ExpCondition(category='R', behavior='Go')]
    """

    def __init__(self, *exp_conds: ExpCondition) -> None:
        if any(not isinstance(cond, ExpCondition) for cond in exp_conds):
            raise TypeError(f"Invalid argument for ExpConditionUnion: {exp_conds} not ExpCondition")
        self.exp_conds = list(exp_conds)

    def to_list(self) -> List[ExpCondition]:
        """Get the list of conditions in the union."""
        return self.exp_conds

    def __iter__(self):
        """Iterate over the conditions in the union, providing each condition one by one."""
        return iter(self.exp_conds)


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
