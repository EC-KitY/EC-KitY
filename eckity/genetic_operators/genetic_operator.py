import random
from abc import abstractmethod

from eckity.event_based_operator import Operator


class GeneticOperator(Operator):
    def __init__(
        self, probability=0.05, arity=0, events=None
    ):
        super().__init__(arity=arity, events=events)
        self.probability = probability

    def apply_operator(self, individuals):
        if random.random() <= self.probability:
            for individual in individuals:
                individual.set_fitness_not_evaluated()
            op_res = self.apply(individuals)
            for ind in op_res:
                ind.applied_operators.append(type(self).__name__)
            return op_res
        return individuals

    @abstractmethod
    def apply(self, individuals):
        pass
