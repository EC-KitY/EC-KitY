from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation

from examples.vectorga.knapsack.knapsack_evaluator import KnapsackEvaluator, NUM_ITEMS

IND_INIT_SIZE = 5


def main():
    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GABitStringVectorCreator(length=NUM_ITEMS),
                      population_size=50,
                      # user-defined fitness evaluation method
                      evaluator=KnapsackEvaluator(),
                      # maximization problem (fitness is prices-weights), so higher fitness is better
                      higher_is_better=True,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.5, k=2),
                          BitStringVectorFlipMutation(probability=0.05)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=500,
        random_seed=64,
        statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()
    algo.best_of_run_.show()


if __name__ == '__main__':
    main()
