from eckity.genetic_operators.mutations.identity_transformation import IdentityTransformation
from eckity.event_based_operator import Operator
from abc import abstractmethod


class FailableOperator(Operator):

    def __init__(self, attempts=5):
        super().__init__()
        self.attempts = attempts

    # TODO add event of on fail or on fail all retries

    def apply_operator(self, payload):
        for i in range(self.attempts):
            (did_not_fail, result) = self.attempt_operator(payload, i)
            if did_not_fail:
                return result
        return self.on_fail(payload)

    # returns tuple of (true if didn't fail, result value)
    #def strict(fun):
    # inspect annotations and check types on call
    #@strict
    @abstractmethod
    def attempt_operator(self, payload, i):
        pass

    # TODO last attempt should perform "on_fail" mechanism
    @abstractmethod
    def on_fail(self, payload):
        pass
