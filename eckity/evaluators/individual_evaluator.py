from overrides import overrides

from eckity.event_based_operator import Operator


class IndividualEvaluator(Operator):

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

		Returns
		-------
		Individual
			the individual with the best fitness out of the given individuals
		"""
		self.applied_individuals = [individual]

	@overrides
	def apply_operator(self, payload):
		return self.evaluate(payload[0], payload[1])
