"""
This module implements the SimpleEvolution class.
"""

from overrides import overrides
from time import time

from eckity.algorithms.algorithm import Algorithm
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.evaluators.simple_population_evaluator import SimplePopulationEvaluator
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker


class SimpleEvolution(Algorithm):

    """
    Simple case evolutionary algorithm.

    Basic evolutionary algorithm that contains one sub-population.
    Does not include and is not meant for multi-objective, co-evolution etc.

    Parameters
    ----------
    population: Population
        The population to be evolved. Contains only one sub-population in simple case.
        Consists of a list of individuals.

    statistics: Statistics or list of Statistics, default=None
        Provide multiple statistics on the population during the evolutionary run.

    breeder: SimpleBreeder, default=SimpleBreeder instance
        Responsible of applying the selection method and operator sequence on the individuals
        in each generation. Applies on one sub-population in simple case.

    population_evaluator: SimplePopulationEvaluator, default=SimplePopulationEvaluator instance
        Responsible of evaluating each individual's fitness concurrently and returns the best individual
        of each subpopulation (returns a single individual in simple case).

    max_generation: int, default=1000
        Maximal number of generations to run the evolutionary process.
        Note the evolution could end before reaching max_generation, depending on the termination checker.

    events: dict(str, dict(object, function)), default=None
        Dictionary of events, where each event holds a dictionary of (subscriber, callback method).

    event_names: list of strings, default=None
        Names of events to publish during the evolution.

    termination_checker: TerminationChecker, default=ThresholdFromTargetTerminationChecker()
        Responsible of checking if the algorithm should finish before reaching max_generation.

    max_workers: int, default=None
        Maximal number of worker nodes for the Executor object that evaluates the fitness of the individuals.

    random_generator: module, default=random
        Random generator module.

    random_seed: float or int, default=current system time
        Random seed for deterministic experiment.

    generation_seed: int, default=None
        Current generation seed. Useful for resuming a previously paused experiment.

    best_of_run_: Individual, default=None
        The individual that has the best fitness in the entire evolutionary run.

    best_of_run_evaluator: IndividualEvaluator, default=None
        The evaluator of the best_of_run individual's sub-population.

    best_of_gen: Individual, default=None
        The individual that has the best fitness in the current generation.

    worst_of_gen: Individual, default=None
        The individual that has the worst fitness in the current generation.

    generation_num: int, default=0
        Current generation number
    """

    def __init__(self,
                 population,
                 statistics=None,
                 breeder=SimpleBreeder(),
                 population_evaluator=SimplePopulationEvaluator(),
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

    def initialize(self):
        super().initialize()
        for stat in self.statistics:
            self.register('after_generation', stat.write_statistics)

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
        self.best_of_gen = self.population_evaluator.act(self.population)

        if self.best_of_gen.better_than(self.best_of_run_):
            self.best_of_run_ = self.best_of_gen

            # TODO maybe it's better for population_evaluator.act to return best of gen and its appropriate evaluator?
            best_of_run_subpopulation = self.population.find_individual_subpopulation(self.best_of_run_)
            self.best_of_run_evaluator = best_of_run_subpopulation.evaluator

        self.worst_of_gen = self.population.sub_populations[0].get_worst_individual()

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

    def finish(self):
        # todo should move to finisher
        self.best_of_run_.show()
        print(self.best_of_run_.get_pure_fitness())

    def get_average_fitness(self):  # TODO check if it should be here or register statistic to breeder or sub pop
        return self.population.get_average_fitness()

    def event_name_to_data(self, event_name):
        if event_name == "init":
            return {"population": self.population,
                    "statistics": self.statistics,
                    "breeder": self.breeder,
                    "termination_checker": self.termination_checker,
                    "max_generation": self.max_generation,
                    "events": self.events,
                    "max_workers": self.max_workers}
        else:
            return {"population": self.population,
                    "best_of_run_": self.best_of_run_,
                    "best_of_gen": self.best_of_gen,
                    "generation_num": self.generation_num}
