from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
import json
import numpy as np
from examples.dnc.DNC_eckity_wrapper import GAIntegerStringVectorCreator
from examples.dnc.dnc_aux import IntVectorUniformMutation, BinPackingEvaluator


def main():
    fitness_dict = {}
    datasets_json = json.load(open('./datasets/hard_parsed.json', 'r'))
    dataset_name = 'BPP_14'
    dataset_item_weights = np.array(datasets_json[dataset_name]['items'])
    dataset_bin_capacity = datasets_json[dataset_name]['max_bin_weight']
    dataset_n_items = len(dataset_item_weights)

    ind_length = dataset_n_items
    min_bound, max_bound = 0, dataset_n_items - 1

    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GAIntegerStringVectorCreator(length=ind_length, bounds=(min_bound, max_bound)),
                      population_size=100,
                      # user-defined fitness evaluation method
                      evaluator=BinPackingEvaluator(n_items=dataset_n_items, item_weights=dataset_item_weights,
                                                    bin_capacity=dataset_bin_capacity, fitness_dict=fitness_dict),
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      # elitism_rate=1 / 300,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.8, k=1),
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
