from eckity.genetic_operators.genetic_operator import GeneticOperator


class IdentityTransformation(GeneticOperator):
    def __init__(self, probability=1.0, events=None):
        """
        Basic mutation operator that does not change the individuals.

        Parameters
        ----------
        probability : float, optional
            Probability to apply each generation, by default 1.0
        events : List[str], optional
            custom events that the operator publishes, by default None
        """
        super().__init__(probability=probability, arity=1, events=events)

    def apply(self, individuals):
        self.applied_individuals = individuals
        return individuals
