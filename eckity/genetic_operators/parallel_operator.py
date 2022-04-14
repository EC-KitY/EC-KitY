from random import choices

from eckity.genetic_operators.mutations.identity_transformation import IdentityTransformation
from eckity.event_based_operator import Operator


class ParallelOperator(Operator):

    def __init__(self, probabilities=None, operators=None):
        super().__init__()
        if operators is None:
            operators = [IdentityTransformation()]
        if probabilities is None:
            probabilities = [0.05]
        self.operators = operators
        self.probabilities = probabilities
        self.chosen_operator = None

    def apply_operator(self, individuals):
        self.chosen_operator = choices(self.operators, self.probabilities)[0]
        self.publish("after_choosing_operator")
        return self.chosen_operator.apply_operator(individuals)

    def get_operator_arity(self):
        # TODO force all operators to have the same num of inds to receive
        self.operators[0].get_operator_arity()

    def event_name_to_data(self, event_name):
        if event_name == "after_choosing_operator":
            return {"chosen_method": self.chosen_operator}
        else:
            return super().event_name_to_data(event_name)

    def __eq__(self, other):
        return super().__eq__(other) \
               and isinstance(other, ParallelOperator) \
               and self.operators == other.operators \
               and self.probabilities == other.probabilities
