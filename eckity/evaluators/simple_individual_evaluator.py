from abc import abstractmethod

from overrides import overrides

from eckity.evaluators.individual_evaluator import IndividualEvaluator


class SimpleIndividualEvaluator(IndividualEvaluator):
	"""
	Computes fitness value for the given individuals.
	In simple case, evaluates each individual separately.
	You will need to extend this class with your user-defined fitness evaluation methods.
	"""

	@overrides
	def evaluate(self, individual, environment_individuals):
		"""
		Updates the fitness score of the given individuals, then returns the best individual

		Parameters
		----------
		individual: Individual
			the current individual to evaluate its fitness

		environment_individuals: list of Individuals
			the individuals in the current individual's environment
			those individuals will affect the current individual's fitness
			(not used in simple case)

		Returns
		-------
		Individual
			the individual with the best fitness out of the given individuals
		"""
		super().evaluate(individual, environment_individuals)
		fitness_score = self._evaluate_individual(individual)
		individual.fitness.set_fitness(fitness_score)
		return individual

	@abstractmethod
	def _evaluate_individual(self, individual):
		"""
		Evaluate the fitness score for the given individual.
		This function must be implemented by subclasses of this class (user-defined evaluators)

		Parameters
		----------
		individual: Individual
			The individual to compute the fitness for

		Returns
		-------
		float
			The evaluated fitness value for the given individual
		"""
		raise ValueError("_evaluate_individual is an abstract method in SimpleIndividualEvaluator")
