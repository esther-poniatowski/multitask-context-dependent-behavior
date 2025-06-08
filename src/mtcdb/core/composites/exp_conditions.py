"""
`core.composites.exp_conditions` [module]

Classes representing the experimental conditions of the behavioral paradigm.

Classes
-------
ExpCondition
"""
from typing import FrozenSet, Mapping, Type, Iterable
from itertools import product

from core.attributes.exp_factors import ExpFactor
from core.composites.attribute_set import AttributeSet, AttributeSetUnion


class ExpCondition(AttributeSet):
    """
    Experimental condition defined by the combination of several experimental factors.

    Usual experimental factors include:

    - task
    - attentional state
    - category of the stimulus
    - behavioral choice
    - response outcome

    Class Attributes
    ----------------
    REQUIRED_FACTORS : Mapping[Type[ExpFactor], FrozenSet[ExpFactor]]
        Experimental factors required to define the experimental condition, along with their allowed
        values.
        Keys: `ExpFactor` subclasses used to define the condition.
        Values: Sets of `ExpFactor` instances, among the valid options for the class.
        In this base class, this attribute is a placeholder and must be defined in subclasses.

    Methods
    -------
    combine
    generate

    Notes
    -----
    All the methods from the `AttributeSet` class are available for the `ExpCondition` class. The
    new behavior of this class restricts the attributes in th set to the *experimental* factors
    only.

    This base class can be instantiated to flexibly define arbitrary experimental conditions.
    Alternatively, it can be inherited to define specific experimental conditions for a given
    pipeline, by enforcing the required factors in the subclass.

    Examples
    --------
    Instantiate an experimental condition with four experimental factors:

    >>> exp_cond = ExpCondition(Task("PTD"), Attention("a"), Category("R"), Behavior("Go"))
    >>> exp_cond
    ExpCondition(task=PTD, attention=a, category=R, behavior=Go)

    Define a subclass for an experimental condition with four factor and their allowed values:

    >>> class SubExpCondition(ExpCondition):
    ...     REQUIRED_FACTORS = {Task: {Task("PTD"), Task("CLK")},
    ...                          Attention: {Attention("a")},
    ...                          Category: {Category("R"), Category("T")},
    ...                          Behavior: {Behavior("Go"), Behavior("NoGo")}}
    ...

    Instantiate a valid experimental condition for the subclass:

    >>> sub_exp_cond = SubExpCondition(Task("PTD"), Attention("a"), Category("R"), Behavior("Go"))

    See Also
    --------
    AttributeSet
    ExpFactor
    """

    REQUIRED_FACTORS: Mapping[Type[ExpFactor], FrozenSet[ExpFactor]]  # placeholder

    def __init__(self, *factors: ExpFactor) -> None:
        """
        Initialize an experimental condition with the provided experimental factors.

        Arguments
        ---------
        *factors : ExpFactor
            Experimental factors defining the condition. Each factor must be an instance of an
            `ExpFactor` subclass and match the required factors for the condition.

        Raises
        ------
        ValueError
            If the factors provided do not match the required factors for the pipeline.
            If some required factors are missing.
        """
        check_required = hasattr(self, "REQUIRED_FACTORS")  # enforce constraints
        for attribute in factors:
            if not isinstance(attribute, ExpFactor):  # check type
                raise TypeError(f"Invalid argument for `ExpCondition`: {attribute} not `ExpFactor`")
            if check_required:
                if type(attribute) not in self.REQUIRED_FACTORS:
                    raise TypeError(
                        f"Invalid type for {self.__class__.__name__}: {type(attribute)}"
                    )
                if attribute not in self.REQUIRED_FACTORS[type(attribute)]:
                    raise ValueError(
                        f"Invalid value for {self.__class__.__name__}: "
                        f"{attribute} not in {self.REQUIRED_FACTORS[type(attribute)]}"
                    )
        if check_required:
            missing = set(self.REQUIRED_FACTORS) - {type(attr) for attr in factors}
            if missing:
                raise ValueError(f"Missing factors for {self.__class__.__name__}: {missing}")
        super().__init__(*factors)  # parent class: `AttributeSet`

    @staticmethod
    def combine(*factors: Iterable[ExpFactor] | ExpFactor) -> AttributeSetUnion["ExpCondition"]:
        """
        Generate experimental conditions by combining a set of experimental factors.

        Arguments
        ---------
        factors : Iterable[ExpFactor] | ExpFactor
            Instances of the experimental factors to consider to generate the experimental
            conditions of interest.

        Returns
        -------
        exp_conditions : AttributeSetUnion[ExpCondition]
            Experimental conditions generated from the Cartesian product of the selected factor
            instances.

        Examples
        --------
        Generate all possible conditions for two tasks, a fixed attentional state, two stimuli and
        both behaviors:

        >>> exp_conditions = ExpCondition.combine_factors(
        ...     tasks=[Task("PTD"), Task("CLK")],
        ...     attentions=Attention("a"),
        ...     categories=[Category("R"), Category("T")],
        ... )
        >>> exp_conditions
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
        sequences = [[fact] if isinstance(fact, ExpFactor) else fact for fact in factors]
        # Generate all combinations of the provided factors
        combinations = product(*sequences)  # List[Tuple[ExpFactor, ...]]
        # Create a list of ExpCondition instances from the combinations
        instances = [ExpCondition(*comb) for comb in combinations]
        return AttributeSetUnion(*instances)

    @classmethod
    def generate(cls) -> AttributeSetUnion["ExpCondition"]:
        """
        Generate all the valid experimental conditions based on the required factors.

        Returns
        -------
        exp_conditions : AttributeSetUnion[ExpCondition]
            All the valid experimental conditions for the pipeline.
        """
        factors = list(cls.REQUIRED_FACTORS.values())
        valid_factors = cls.combine(*factors)  # list of ExpCondition
        return valid_factors
