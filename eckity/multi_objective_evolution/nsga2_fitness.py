from typing import List

from overrides import overrides

from eckity.fitness.fitness import Fitness
from eckity.individual import Individual
import random as rd


class NSGA2Fitness(Fitness):
	def __init__(self,
				 fitness: List[float] = None,
				 higher_is_better=False):
		is_evaluated = fitness is not None
		super().__init__(higher_is_better=higher_is_better, is_evaluated=is_evaluated)
		self.fitness: List[float] = fitness  # list of objectivs
		self.crowding = 0
		self.front_rank = float("inf")
		if self.fitness!=None and type(self.higher_is_better) is bool:
			self.higher_is_better = [self.higher_is_better] * len(fitness)



	def set_fitness(self, fitness):
		if self._is_evaluated:
			raise AttributeError('fitness already evaluated and set to', self.fitness)
		self.fitness = fitness
		self._is_evaluated = True
		if type(self.higher_is_better) is bool:
			self.higher_is_better = [self.higher_is_better] * len(fitness)

	@overrides
	def get_pure_fitness(self):
		if not self._is_evaluated:
			raise ValueError('Fitness not evaluated yet')
		return self.fitness

	@overrides
	def set_not_evaluated(self):
		self._is_evaluated = False
		self.fitness = None
		self.crowding = 0
		self.front_rank = float("inf")

	def check_comparable_fitnesses(self, other_fitness: Fitness, ind: Individual, other_ind: Individual):
		if not isinstance(other_fitness, NSGA2Fitness):
			raise TypeError('Expected NSGA2Fitness object in better_than, got', type(other_fitness))
		if not self.is_fitness_evaluated() or not other_fitness.is_fitness_evaluated():
			raise ValueError('Fitnesses must be evaluated before comparison')
		if len(other_fitness.get_augmented_fitness(other_ind)) != len(self.get_augmented_fitness(ind)):
			raise ValueError('Fitnesses must be of the same lngth')

	def better_than(self, ind, other_fitness, other_ind):
		'''

		Parameters
		----------
		ind:Individual
		other_fitness:NSGA2Fitness
		other_ind:Individual

		Returns : True ind has lower rank or equal rank and bigger crwoding value
		-------

		'''
		if self.front_rank == float("inf"):  # first iteration
			return bool(rd.getrandbits(1))  # random true false
		else:
			self.check_comparable_fitnesses(other_fitness, ind, other_ind)
			if self.front_rank == other_ind.fitness.front_rank:
				return self.crowding > other_ind.fitness.crowding
			return self.front_rank > other_ind.fitness.front_rank

	def equal_to(self, ind, other_fitness, other_ind):
		return self.front_rank == other_ind.fitness.front_rank and \
			   self.crowding == other_ind.fitness.crowding

	def dominate(self, ind, other_fitness, other_ind):
		self.check_comparable_fitnesses(other_fitness, ind, other_ind)
		self_fit = self.get_augmented_fitness(ind)
		other_fit = other_fitness.get_augmented_fitness(other_ind)
		return all([self._o1_at_least_o2(o1, o2, h_is_b) for h_is_b, o1, o2 in
					zip(self.higher_is_better, self_fit, other_fit)]) and \
			   any([self._o1_better_then_o2(o1, o2, h_is_b) for h_is_b, o1, o2 in
					zip(self.higher_is_better, self_fit, other_fit)])

	def _o1_at_least_o2(self, o1, o2, higher_is_better):
		'''

		Parameters
		----------
		o1: objective i of individual 1
		o2: objective i of individual 2
		higher_is_better :boolean - is higher better for objective i of he fitness ?

		Returns :x1 >= x2 or x1 <= x2 according to higher is better
		-------

		'''
		if higher_is_better:
			return o1 >= o2
		else:
			return o1 <= o2

	def _o1_better_then_o2(self, o1, o2, higher_is_better):
		'''

		Parameters
		----------
		o1: objective i of individual 1
		o2: objective i of individual 2
		higher_is_better :boolean - is higher better for objective i of he fitness ?

		Returns :x1 > x2 or x1 < x2 according to higher is better
		-------

		'''
		if higher_is_better:
			return o1 > o2
		else:
			return o1 < o2

	# def equal_to(self, ind, other_fitness, other_ind):
	#     self.check_comparable_fitnesses(other_fitness,ind,other_ind)
	#     self_fit = self.get_augmented_fitness(ind)
	#     other_fit = other_fitness.get_augmented_fitness(other_ind)
	#     all([x1 == x2 for x1, x2 in zip(self_fit, other_fit)])

	def __getstate__(self):
		state = self.__dict__.copy()
		if not self.should_cache_between_gens:
			state['_is_evaluated'] = False
			state['fitness'] = None
		return state
