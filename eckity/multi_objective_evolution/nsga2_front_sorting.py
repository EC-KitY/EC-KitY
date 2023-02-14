from collections import defaultdict
from typing import List

from eckity.individual import Individual
from eckity.population import Population


class NSGA2FrontSorting():
	'''
    This class is incharge of setitng the value of selecting only the best individuals
    out of the population
    (mening that the are dominated by as littel amount of other Individuals as posible)

    this class allso set the values of the "front_rank" and "crowding" of each individual
    '''

	def select_for_population(self, population: Population, new_pop_size=None):
		for sub_population in population.sub_populations:
			dest_inds = []
			dest_inds = self._select(sub_population.individuals, dest_inds, new_pop_size)
			sub_population.individuals = dest_inds

	def _select(self, source_inds: List[Individual], dest_inds: List[Individual], pop_size: int):
		'''

        Parameters
        ----------
        source_inds : list of individuals (sub_population.individuals)
        dest_inds: the pareto front
        pop_size :  the size we destend the pareto front to be

        Returns :  the pareto front
        -------

        '''
		if not pop_size:
			pop_size = len(source_inds) // 2

		front_rank = 1
		self._init_domination_dict(source_inds)
		while len(dest_inds) < pop_size:
			new_pareto_front = self._pareto_front_finding(source_inds)
			self._calc_fronts_crowding(new_pareto_front)
			self._update_new_pareto_front_rank(new_pareto_front, front_rank)

			total_pareto_size = len(new_pareto_front) + len(dest_inds)
			if total_pareto_size > pop_size:
				number_solutions_needed = pop_size - len(dest_inds)
				new_pareto_front.sort(key=lambda x: x.fitness.crowding, reverse=True)
				new_pareto_front = new_pareto_front[
								   :int(number_solutions_needed)]  # take the individuals with the largest crowding
			dest_inds += new_pareto_front
			source_inds = self._remove_pareto_front(source_inds, new_pareto_front)
			front_rank += 1
		return dest_inds

	def _remove_pareto_front(self, source_inds, pareto_front):
		for dominating_ind in pareto_front:
			for dominated_ind in self.domination_dict[dominating_ind].dominates:
				self.domination_dict[dominated_ind].domination_counter -= 1
		return list(set(source_inds) - set(pareto_front))

	def _update_new_pareto_front_rank(self, new_pareto_front: List[Individual], front_rank: int):
		for ind in new_pareto_front:
			ind.fitness.front_rank = front_rank

	def _calc_fronts_crowding(self, front: List[Individual]):
		for ind in front:
			ind.fitness.crowding = 0

		objectiv_indexes = range(len(front[0].get_pure_fitness()))
		for objective_index in objectiv_indexes:
			front.sort(key=lambda x: x.get_pure_fitness()[objective_index])  # sort for each objectiv
			front[0].fitness.crowding = float("inf")
			front[-1].fitness.crowding = float("inf")
			for i in range(1, len(front) - 1):
				curr_crowding = front[i + 1].get_pure_fitness()[objective_index] - front[i - 1].get_pure_fitness()[
					objective_index]
				curr_crowding /= (front[-1].get_pure_fitness()[objective_index] - front[0].get_pure_fitness()[
					objective_index])
				front[i].fitness.crowding += curr_crowding

	def _init_domination_dict(self, source_inds: List[Individual]):
		self.domination_dict = defaultdict(lambda: DominationCounter())
		for i, ind1 in enumerate(source_inds):
			for ind2 in source_inds[i + 1:]:
				self._habdle_domination(ind1, ind2)

	def _habdle_domination(self, ind1, ind2):
		if ind2.fitness.dominate(ind2, ind1.fitness, ind1):
			self._increase_domination_counter(ind2, ind1)
		elif ind1.fitness.dominate(ind1, ind2.fitness, ind2):
			self._increase_domination_counter(ind1, ind2)

	def _increase_domination_counter(self, dominating, dominated):
		self.domination_dict[dominating].dominates.append(dominated)
		self.domination_dict[dominated].domination_counter += 1

	def _pareto_front_finding(self, source_inds: List[Individual]):
		pareto_front = [ind for ind in source_inds if self.domination_dict[ind].domination_counter == 0]
		return pareto_front


class DominationCounter:
	def __init__(self):
		'''
        self.dominates : a list of all the individuals this individual dominates
        self.domination_counter : a counter of how many other individuals dominates this one
        '''
		self.dominates = []
		self.domination_counter = 0
