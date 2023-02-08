import math
import time

from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.multi_objective_evolution.moe_evolution import MOEvolution
from eckity.multi_objective_evolution.moe_breeder import MOEBreeder
from eckity.creators.ga_creators.moe_vector_creator import MOEitStringVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.multi_objective_evolution.moe_fitness import MOEFitness
from eckity.population import Population
from eckity.statistics.minimal_print_statistics import MinimalPrintStatistics
from eckity.subpopulation import Subpopulation
from eckity.genetic_encodings.ga.float_vector import FloatVector

from eckity.termination_checkers.iteration_termination_checker import IterationTerminationChecker


class Zdt1Evaluator(SimpleIndividualEvaluator):
	def _evaluate_individual(self, individual):
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
		k = len(individual.vector)
		f1 = individual.vector[0]
		g = 1 + (9 / (k - 1)) * sum(individual.vector[1:])
		f2 = 1 - math.sqrt(f1 / g)
		return [f1, f2]


def main():
	# Initialize the evolutionary algorithm
	algo = MOEvolution(
		Population([Subpopulation(
			creators=GAVectorCreator(length=3, bounds=(0, 1), fitness_type=MOEFitness, vector_type=FloatVector),
			population_size=150,
			# user-defined fitness evaluation method
			evaluator=Zdt1Evaluator(),
			# maximization problem (fitness is sum of values), so higher fitness is better
			higher_is_better=False,
			elitism_rate=1 / 300,
			# genetic operators sequence to be applied in each generation
			operators_sequence=[
				VectorKPointsCrossover(probability=0.7, k=1),
				FloatVectorUniformNPointMutation(probability=0.3, n=3)  # maybe chnge mutation
			],
			selection_methods=[
				# (selection method, selection probability) tuple
				(TournamentSelection(tournament_size=3, higher_is_better=True), 1)
			]
		)]),
		breeder=MOEBreeder(),
		max_workers=4,
		max_generation=10,

		termination_checker=IterationTerminationChecker(),
		statistics=MinimalPrintStatistics()
	)

	# evolve the generated initial population
	start_time = time.time()
	algo.evolve()
	total_time = time.time() - start_time
	print(f"the total time is :{total_time}")

	# Execute (show) the best solution
	print(algo.execute())


if __name__ == '__main__':
	main()