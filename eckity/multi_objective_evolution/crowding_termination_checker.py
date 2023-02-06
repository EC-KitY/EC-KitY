from eckity.termination_checkers.termination_checker import TerminationChecker


class CrowdingTerminationChecker(TerminationChecker):
	"""
	Concrete Termination Checker that checks the distance from best existing fitness value to target fitness value.

	Parameters
	----------
	threshold: float, default=0.1
		what is the maximal value of crowding that should be allowed in he population
	"""

	def __init__(self, threshold=0.1):
		super().__init__()
		self.threshold = threshold

	def should_terminate(self, population=None, best_individual=None, gen_number=None):
		"""
		allawys return false so the program wont terminate until it finishes its iteration
		Parameters
		----------
		population: Population
			The evolutionary experiment population of individuals.

		best_individual: Individual
			The individual that has the best fitness of the current generation.

		gen_number: int
			Current generation number.

		Returns
		-------
		bool
			True if the algorithm should terminate early, False otherwise.
		"""
		max_crwding_in_pop = self._find_max_crowding(population)
		return max_crwding_in_pop < self.threshold and max_crwding_in_pop != 0

	def _find_max_crowding(self, population):
		crowdings = []
		for sup_pop in population.sub_populations:
			crowdings.append(
				max([ind.fitness.crowding for ind in sup_pop.individuals if ind.fitness.crowding != float("inf")]))
		return max(crowdings)
