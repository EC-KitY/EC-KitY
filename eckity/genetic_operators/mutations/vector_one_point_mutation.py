from random import choices

from eckity.genetic_operators.failable_operator import FailableOperator
from eckity.genetic_operators.genetic_operator import GeneticOperator


class VectorNPointMutation(FailableOperator):
    def __init__(self, n=1, probability=1, arity=1, mut_val_getter=None, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
        if mut_val_getter is None:
            mut_val_getter = self.default_mut_val_getter
        self.default_mut_val_gen = mut_val_getter
        self.n = n

    @staticmethod
    def default_mut_val_getter(vec, idx):
        return vec.get_random_number_in_bounds(vec, idx)

    def apply(self, individuals):
        for individual in individuals:
            # randomly select n points of the vector (without repetitions)
            m_points = choices(individual.get_vector(), k=self.n)
            # obtain the mutated values
            mut_vals = [self.default_mut_val_gen(individual, m_point) for m_point in m_points]

            # update the mutated value in-place
            for m_point, mut_val in zip(m_points, mut_vals):
                individual.set_cell_value(m_point, mut_val)

        self.applied_individuals = individuals
        return individuals
