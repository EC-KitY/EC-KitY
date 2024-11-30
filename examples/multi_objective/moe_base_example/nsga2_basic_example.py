import math
import time

from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.multi_objective_evolution.nsga2_fitness import NSGA2Fitness
from eckity.multi_objective_evolution.nsga2_plot import NSGA2Plot
from eckity.multi_objective_evolution.nsga2_evolution import NSGA2Evolution
from eckity.multi_objective_evolution.nsga2_breeder import NSGA2Breeder
from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)
from eckity.genetic_operators.crossovers.vector_k_point_crossover import (
    VectorKPointsCrossover,
)
from eckity.genetic_operators.mutations.vector_random_mutation import (
    FloatVectorUniformNPointMutation,
)
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)

from eckity.population import Population
from eckity.statistics.minimal_print_statistics import MinimalPrintStatistics
from eckity.multi_objective_evolution.moe_best_worst_statistics import (
    MOEBestWorstStatistics,
)
from eckity.subpopulation import Subpopulation
from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.multi_objective_evolution.crowding_termination_checker import (
    CrowdingTerminationChecker,
)


import random

random.seed(0)


class NSGA2BasicExampleEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        """
        Compute the fitness value of a given individual.

        Parameters
        ----------
        individual: Vector
            The individual to compute the fitness value for.

        Returns
        -------
        list
            The evaluated fitness value for each of the objectives of the given individual.
        """
        n = len(individual.vector)
        fit1 = 1 - math.exp(
            -sum([(x - 1 / math.sqrt(n)) ** 2 for x in individual.vector])
        )
        fit2 = 1 - math.exp(
            -sum([(x + 1 / math.sqrt(n)) ** 2 for x in individual.vector])
        )
        return [fit1, fit2]


def main():
    # Initialize the evolutionary algorithm
    algo = NSGA2Evolution(
        Population(
            [
                Subpopulation(
                    creators=GAVectorCreator(
                        length=3,
                        bounds=(-4, 4),
                        fitness_type=NSGA2Fitness,
                        vector_type=FloatVector,
                    ),
                    population_size=150,
                    # user-defined fitness evaluation method
                    evaluator=NSGA2BasicExampleEvaluator(),
                    # maximization problem (fitness is sum of values), so higher fitness is better
                    higher_is_better=False,
                    elitism_rate=1 / 300,
                    # genetic operators sequence to be applied in each generation
                    operators_sequence=[
                        VectorKPointsCrossover(probability=0.7, k=1),
                        FloatVectorUniformNPointMutation(probability=0.3, n=3),
                    ],
                    selection_methods=[
                        # (selection method, selection probability) tuple
                        (
                            TournamentSelection(
                                tournament_size=3, higher_is_better=True
                            ),
                            1,
                        )
                    ],
                )
            ]
        ),
        breeder=NSGA2Breeder(),
        max_workers=4,
        executor="process",
        max_generation=150,
        termination_checker=CrowdingTerminationChecker(0.01),
        statistics=[
            MOEBestWorstStatistics(),
            MinimalPrintStatistics(),
        ],
    )
    ploter = NSGA2Plot()
    algo.register("evolution_finished", ploter.print_plots)
    # evolve the generated initial population
    start_time = time.time()
    algo.evolve()
    total_time = time.time() - start_time
    print(f"the total time is :{total_time}")

    # Execute (show) the best solution
    print(algo.execute())


if __name__ == "__main__":
    main()
