from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
import json
import numpy as np
from DNC_eckity_wrapper import DeepNeuralCrossoverConfig, GAIntegerStringVectorCreator, DeepNeuralCrossover
from random import random


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


def main():
    fitness_dict = {}
    datasets_json = json.load(open('./datasets/hard_parsed.json', 'r'))
    dataset_name = 'BPP_14'
    dataset_item_weights = np.array(datasets_json[dataset_name]['items'])
    dataset_bin_capacity = datasets_json[dataset_name]['max_bin_weight']
    dataset_n_items = len(dataset_item_weights)

    ind_length = dataset_n_items
    min_bound, max_bound = 0, dataset_n_items - 1
    population_size = 100

    individual_creator = GAIntegerStringVectorCreator(length=ind_length, bounds=(min_bound, max_bound))
    bpp_eval = BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                   bin_capacity=dataset_bin_capacity, fitness_dict=fitness_dict)

    dnc_config = DeepNeuralCrossoverConfig(
        embedding_dim=64,
        sequence_length=ind_length,
        num_embeddings=dataset_n_items + 1,
        running_mean_decay=0.95,
        batch_size=1024,
        learning_rate=1e-4,
        use_device='cpu',
        n_parents=2,
        epsilon_greedy=0.3
    )

    dnc_op = DeepNeuralCrossover(probability=0.8, population_size=population_size, dnc_config=dnc_config,
                                 individual_evaluator=bpp_eval, vector_creator=individual_creator)

    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=individual_creator,
                      population_size=population_size,
                      # user-defined fitness evaluation method
                      evaluator=bpp_eval,
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      # elitism_rate=1 / 300,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          dnc_op,
                          IntVectorUniformMutation(probability=0.5, probability_for_each=0.1)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=5, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=6000,
        # termination_checker=ThresholdFromTargetTerminationChecker(optimal=100, threshold=0.0),
        statistics=BestAverageWorstStatistics(), random_seed=4242
    )

    # evolve the generated initial population
    algo.evolve()

    # Execute (show) the best solution
    print(algo.execute())


if __name__ == '__main__':
    main()
