from time import time
from overrides import overrides

from eckity.algorithms.algorithm import Algorithm
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.evaluators.simple_population_evaluator import SimplePopulationEvaluator
from eckity.multi_objective_evolution.nsga2_front_sorting import NSGA2FrontSorting

from eckity.termination_checkers.threshold_from_target_termination_checker \
	import ThresholdFromTargetTerminationChecker


class NSGA2Evolution(Algorithm):
	def __init__(self,
				 population,
				 statistics=None,
				 breeder=SimpleBreeder(),
				 population_evaluator=SimplePopulationEvaluator(),
				 NSGA2FrontSorting=NSGA2FrontSorting(),
				 max_generation=500,
				 events=None,
				 event_names=None,
				 termination_checker=ThresholdFromTargetTerminationChecker(threshold=0),
				 max_workers=None,
				 random_generator=None,
				 random_seed=time(),
				 generation_seed=None,
				 best_of_run_=None,
				 best_of_run_evaluator=None,
				 best_of_gen=None,
				 worst_of_gen=None,
				 generation_num=0):

		if event_names is None:
			_event_names = ['before_eval', 'after_eval', 'before_breeding', 'after_breeding']
		else:
			_event_names = event_names

		if statistics is None:
			statistics = []

		super().__init__(population, statistics=statistics, breeder=breeder, population_evaluator=population_evaluator,
						 events=events, event_names=_event_names, max_workers=max_workers,
						 random_generator=random_generator, random_seed=random_seed, generation_seed=generation_seed,
						 termination_checker=termination_checker, generation_num=generation_num)

		self.termination_checker = termination_checker
		self.best_of_run_ = best_of_run_
		self.best_of_run_evaluator = best_of_run_evaluator
		self.best_of_gen = best_of_gen
		self.worst_of_gen = worst_of_gen
		self.max_generation = max_generation

		self.final_generation_ = None
		self.NSGA2FrontSorting = NSGA2FrontSorting

	#
	@overrides
	def generation_iteration(self, gen):
		"""
		Performs one iteration of the evolutionary run, for the current generation

		Parameters
		----------
		gen:
			current generation number (for example, generation #100)

		Returns
		-------
		None.
		"""

		# breed population
		self.breeder.breed(self.population)
		# Evaluate the entire population and get the best individual

		self.population_evaluator.act(self.population)

		self.NSGA2FrontSorting.select_for_population(self.population)
		self.best_of_gen = self._get_pareto_fronts()

	def _get_pareto_fronts(self):
		'''
		Returns: the pareto front for each sub_population
		-------
		'''
		pareto_fronts = []
		for sub_pop in self.population.sub_populations:
			pareto_fronts.append([ind for ind in sub_pop.individuals if ind.fitness.front_rank == 1])
		return pareto_fronts

	def initialize(self):
		"""
		Initialize the evolutionary algorithm

		Register statistics to `after_generation` event
		"""
		super().initialize()
		for stat in self.statistics:
			self.register('after_generation', stat.write_statistics)

	def execute(self, **kwargs):
		"""
		Compute output using best evolved individual.
		Use `execute` in a non-sklearn setting.
		Input keyword arguments that set variable values.
		For example if `terminal_set=['x', 'y', 1, -1]` then call `execute(x=..., y=...)`.

		Parameters
		----------
		**kwargs : keyword arguments
			The input to the program (tree).

		Returns
		-------
		object
			Output as computed by the best individual of the evolutionary run.

		"""
		return self.best_of_run_.execute(**kwargs)

	@overrides
	def finish(self):
		"""
		Finish the evolutionary run by showing the best individual and printing the best fitness
		"""
		self.best_of_run_.show()
		print(self.best_of_run_.get_pure_fitness())

	def event_name_to_data(self, event_name):
		if event_name == "init":
			return {
				"population": self.population,
				"statistics": self.statistics,
				"breeder": self.breeder,
				"termination_checker": self.termination_checker,
				"max_generation": self.max_generation,
				"events": self.events,
				"max_workers": self.max_workers
			}

		# default case
		return {
			"population": self.population,
			"best_of_run_": self.best_of_run_,
			"best_of_gen": self.best_of_gen,
			"generation_num": self.generation_num
		}
