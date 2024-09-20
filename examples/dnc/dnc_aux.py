from random import random

import numpy as np

from eckity.evaluators import SimpleIndividualEvaluator
from eckity.genetic_operators import VectorNPointMutation


def uniform_cell_selector(vec):
    return list(range(vec.size()))


class IntVectorUniformMutation(VectorNPointMutation):
    """
    Uniform N Point Integer Mutation
    """

    def __init__(self, probability=0.5, arity=1, events=None, probability_for_each=0.1):
        self.probability_for_each = probability_for_each
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(
                             index) if random() <= self.probability_for_each else individual.cell_value(index),
                         events=events, cell_selector=uniform_cell_selector)


class BinPackingEvaluator(SimpleIndividualEvaluator):

    def __init__(self, n_items, item_weights, bin_capacity, fitness_dict):
        super().__init__()
        self.n_items = n_items
        self.item_weights = item_weights
        self.bin_capacity = bin_capacity
        self.fitness_dict = fitness_dict

    def evaluate_individual(self, individual):
        """
            Compute the fitness value of a given individual.

            Parameters
            ----------
            individual: Vector
                The individual to compute the fitness value for.

            Returns
            -------
            float
                The evaluated fitness value of the given individual.
        """
        return self.get_bin_packing_fitness(np.array(individual.vector))

    def get_bin_packing_fitness(self, individual, penalty=100):
        fitness_dict = self.fitness_dict

        if tuple(individual) in fitness_dict:
            return fitness_dict[tuple(individual)]

        fitness = 0
        bin_capacities = np.zeros(self.n_items)
        legal_solution = True

        for item_index, bin_index in enumerate(individual):
            bin_capacities[bin_index] += self.item_weights[item_index]

            if bin_capacities[bin_index] > self.bin_capacity:
                legal_solution = False
                fitness -= penalty

        if legal_solution:
            utilized_bins = bin_capacities[bin_capacities > 0]
            fitness = ((bin_capacities / self.bin_capacity) ** 2).sum() / len(utilized_bins)

        fitness_dict[tuple(individual)] = fitness
        return fitness
