from eckity.genetic_operators.mutations.identity_transformation import IdentityTransformation
from eckity.event_based_operator import Operator
from abc import abstractmethod


class FailableOperator(Operator):

    def __init__(self, retries=4):
        super().__init__()
        self.retries = retries

    # TODO add event of on fail or on fail all retries

    def apply_operator(self, payload):
        for i in range(self.retries):
            (did_not_fail, result) = self.apply_operator_without_fail(payload, i)
            if did_not_fail:
                return result
        return on_fail(payload)

    # returns tuple of (true if didn't fail, result value)
    #def strict(fun):
    # inspect annotations and check types on call
    #@strict
    @abstractmethod
    def apply_operator_without_fail(self, payload, i):
        pass

    @abstractmethod
    def on_fail(self, payload):
        pass
