from copy import deepcopy

from eckity.fitness.fitness import Fitness


class Individual:
    id = 1

    def __init__(self, fitness: Fitness):
        self.update_id()
        self.gen = 0
        self.fitness = fitness

        # informational only
        self.cloned_from = []  # chain of ids from gen 0
        self.selected_by = []  # last gen
        self.applied_operators = []  # last gen


    def update_id(self):
        self.id = Individual.id
        Individual.id += 1

    def set_fitness_not_evaluated(self):
        self.fitness.set_not_evaluated()

    def clone(self):
        result = deepcopy(self)
        result.cloned_from.append(self.id)
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
