from copy import deepcopy

from eckity.fitness.fitness import Fitness


class Individual:
    def __init__(self, fitness: Fitness):
        self.fitness = fitness

    def set_fitness_not_evaluated(self):
        self.fitness.set_not_evaluated()

    def clone(self):
        return deepcopy(self)

    def get_pure_fitness(self):
        return self.fitness.get_pure_fitness()

    def get_augmented_fitness(self):
        return self.fitness.get_augmented_fitness(self)

    def better_than(self, other):
        if other is None:
            return True

        return self.fitness.better_than(self, other.fitness, other)
