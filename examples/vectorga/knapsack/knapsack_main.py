from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker

import random

from examples.vectorga.knapsack.knapsack_evaluator import KnapsackEvaluator

NUM_ITEMS = 20
IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50


def main():
    # Generate ramdom items for the problem
    items = {random.randint(1, 10): random.uniform(0, 100) for _ in range(NUM_ITEMS)}

    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GABitStringVectorCreator(length=5),
                      population_size=50,
                      # user-defined fitness evaluation method
                      evaluator=KnapsackEvaluator(),
                      # minimization problem (fitness is MAE), so higher fitness is worse
                      higher_is_better=False,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.5, k=2),
                          BitStringVectorFlipMutation(probability=0.05)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=3, higher_is_better=False), 1)
                      ]),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=500,
        random_seed=64,
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=0, threshold=0.001),
        statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()

    # execute the best individual after the evolution process ends
    print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')


if __name__ == '__main__':
    main()
