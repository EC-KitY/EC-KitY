from random import choices

from eckity.genetic_operators.failable_operator import FailableOperator
from eckity.genetic_operators.genetic_operator import GeneticOperator


class VectorNPointMutation(FailableOperator):
    def __init__(self, n=1, probability=1, arity=1, mut_val_getter=None,
                 success_checker=None, on_fail=None, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
        self.n = n

        if on_fail is None:
            on_fail = self.default_on_fail
        self.on_fail = on_fail

        if success_checker is None:
            success_checker = self.default_success_checker
        self.success_checker = success_checker

        if mut_val_getter is None:
            mut_val_getter = self.default_mut_val_getter
        self.mut_val_getter = mut_val_getter

    @staticmethod
    def default_on_fail(vectors):
        return [vector.replace_vector_part_random([vector.get_random_number_in_bounds(0)]) for vector in vectors]

    @staticmethod
    def default_mut_val_getter(vec, idx):
        return vec.get_random_number_in_bounds(vec, idx)

    @staticmethod
    def default_success_checker(old_vec, new_vec):
        return new_vec.check_if_in_bounds()

    def attempt_operator(self, individuals, attempt_num):
        succeeded = True
        for individual in individuals:
            old_individual = individual.clone()

            # randomly select n points of the vector (without repetitions)
            m_points = choices(individual.get_vector(), k=self.n)
            # obtain the mutated values
            mut_vals = [self.mut_val_getter(individual, m_point) for m_point in m_points]

            # update the mutated value in-place
            for m_point, mut_val in zip(m_points, mut_vals):
                individual.set_cell_value(m_point, mut_val)

            if not self.success_checker(old_individual, individual):
                succeeded = False
                break

        if not succeeded:
            self.on_fail(individuals)

        self.applied_individuals = individuals
        return succeeded, individuals
