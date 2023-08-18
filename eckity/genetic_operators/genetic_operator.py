from abc import abstractmethod
from random import uniform

from eckity.event_based_operator import Operator


class GeneticOperator(Operator):
    def __init__(self, probability=0.05, arity=0, events=None):
        super().__init__(events=events, arity=arity)
        self.probability = probability

    def apply_operator(self, individuals):
        if uniform(0, 1) <= self.probability:
            update_parents = individuals[0].update_parents
            if update_parents:
                parents = [ind.clone() for ind in individuals]

            for individual in individuals:
                individual.set_fitness_not_evaluated()

            op_res = self.apply(individuals)
            for ind in op_res:
                ind.applied_operators.append(type(self).__name__)

                if update_parents:
                    ind.parents.extend(parents)
            return op_res
        return individuals

    @abstractmethod
    def apply(self, individuals):
        pass
