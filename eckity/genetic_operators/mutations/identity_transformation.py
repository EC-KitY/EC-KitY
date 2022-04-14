from eckity.genetic_operators.probabilistic_condition_operator import ProbabilisticConditionOperator


class IdentityTransformation(ProbabilisticConditionOperator):
    def __init__(self,probability=1, events=None):
        super().__init__(probability=probability, arity=1, events=events)

    def apply(self, individuals):
        self.applied_individuals = individuals
        return individuals
