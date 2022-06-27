
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorFlipMutation, \
    BitStringVectorNFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker


class OneMaxEvaluator(SimpleIndividualEvaluator):
    def _evaluate_individual(self, individual):
        return sum(individual.vector)


def main():
    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GABitStringVectorCreator(length=100),
                      population_size=300,
                      # user-defined fitness evaluation method
                      evaluator=OneMaxEvaluator(),
                      # minimization problem (fitness is MAE), so higher fitness is worse
                      higher_is_better=True,
                      elitism_rate=1/300,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.7, k=1),
                          BitStringVectorNFlipMutation(probability=1, probability_for_each=0.1, n=100)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=3, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=500,
        random_seed=64,
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=100, threshold=0.001),
        statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()

    # execute the best individual after the evolution process ends
    # print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')
    algo.best_of_run_.show()


if __name__ == '__main__':
    main()
