from copy import deepcopy

from eckity.fitness.fitness import Fitness


class Individual:
    """
    A candidate solution to the problem.
    This class is abstract and should not be instansiated directly.

    Parameters
    ----------
    fitness: Fitness
        This object manages the fitness state of the individual.

    Attributes
    ----------
    id: int
        The unique id of the individual.
    gen: int
        The generation number of the individual.
    fitness: Fitness
        This object manages the fitness state of the individual.
    cloned_from: list
        A list of ids of individuals that this individual was cloned from.
    selected_by: list
        A list of selection methods that selected this individual in
        the last generation.
    applied_operators: list
        A list of genetic operators that were applied on this individual
        in the last generation.
        *** Note that failed operators are still included in this list. ***
    """

    id = 1

    def __init__(self, fitness: Fitness, update_parents: bool = False):
        self.update_id()
        self.gen = 0
        self.fitness = fitness

        # informational only
        self.cloned_from = []  # chain of ids from gen 0
        self.selected_by = []  # last gen
        self.applied_operators = []  # last gen

        self.update_parents = update_parents
        if update_parents:
            self.parents = []  # last gen

    def update_id(self):
        self.id = Individual.id
        Individual.id += 1

    def set_fitness_not_evaluated(self):
        self.fitness.set_not_evaluated()

    def clone(self):
        result = deepcopy(self)
        result.cloned_from.append(self.id)
        if result.update_parents:
            result.parents = []
        result.update_id()
        return result

    def get_pure_fitness(self):
        return self.fitness.get_pure_fitness()

    def get_augmented_fitness(self):
        return self.fitness.get_augmented_fitness(self)

    def better_than(self, other):
        if other is None:
            return True

        return self.fitness.better_than(self, other.fitness, other)
