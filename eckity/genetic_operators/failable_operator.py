from eckity.genetic_operators.genetic_operator import GeneticOperator
from eckity.genetic_operators.mutations.identity_transformation import IdentityTransformation
from eckity.event_based_operator import Operator
from abc import abstractmethod


class FailableOperator(GeneticOperator):

    def __init__(self, probability=0.05, arity=0, events=None, attempts=5):
        super().__init__(probability, arity, events)
        if attempts <= 1:
            raise ValueError("Number of attempts must be at least 1")
        self.attempts = attempts

    # TODO add event of on fail or on fail all retries
    def apply(self, payload):
        result = None
        for i in range(self.attempts):
            succeeded, result = self.attempt_operator(payload, i)
            if succeeded:
                break
        return result

    # returns tuple of (succeeded or not, result value)
    @abstractmethod
    def attempt_operator(self, payload, i):
        pass
