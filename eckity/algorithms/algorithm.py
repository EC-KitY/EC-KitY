"""
This module implements the Algorithm class.
"""

import logging
import sys
from abc import ABC, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from time import time
from typing import Any, Callable, Dict, List, Union

from overrides import overrides

from eckity.population import Population
from eckity.subpopulation import Subpopulation
from eckity.breeders import Breeder
from eckity.evaluators import PopulationEvaluator
from eckity.event_based_operator import Operator
from eckity.individual import Individual
from eckity.random import RNG
from eckity.statistics.statistics import Statistics
from eckity.termination_checkers import TerminationChecker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class Algorithm(Operator, ABC):
    """
    Evolutionary algorithm to be executed.

    Abstract Algorithm that can be extended to concrete algorithms,
    such as SimpleEvolution, Coevolution etc.

    Parameters
    ----------
    population: Population
        The population to be evolved.
        Consists of several sub-populations.
        ref: https://api.eckity.org/eckity/population.html

    statistics: Statistics or list of Statistics, default=None
        Provide statistics on the population during the evolution.
        ref: https://api.eckity.org/eckity/statistics.html

    breeder: Breeder, default=SimpleBreeder()
        Responsible for applying selection and operator sequence on individuals
        in each generation. Applies on one sub-population in simple case.
        ref: https://api.eckity.org/eckity/breeders.html

    population_evaluator: PopulationEvaluator,
                          default=SimplePopulationEvaluator()
        Evaluates individual fitness scores concurrently and returns the best
        individual of each subpopulation (one individual in simple case).
        ref: https://api.eckity.org/eckity/evaluators.html

    max_generation: int, default=100
        Maximal number of generations to run the evolutionary process.
        Note the evolution could end before reaching max_generation,
        depends on the termination checker.
        Note there are up to `max_generation + 1` fitness calculations,
        but only up to `max_generation` selections

    events: dict(str, dict(object, function)), default=None
        dict of events, each event holds a dict (subscriber, callback).

    event_names: list of strings, default=None
        Names of events to publish during the evolution.

    termination_checker: TerminationChecker or a list of TerminationCheckers,
                          default=None
        Checks if the algorithm should terminate early.
        ref: https://api.eckity.org/eckity/termination_checkers.html

    max_workers: int, default=None
        Maximal number of worker nodes for the Executor object
        that evaluates the fitness of the individuals.
        ref: https://docs.python.org/3/library/concurrent.futures.html

    random_generator: RNG, default=RNG()
        Random Number Generator.

    random_seed: int, default=current system time
        Random seed for deterministic experiment.

    generation_seed: int, default=None
        Current generation seed.
        Useful for resuming a previously paused experiment.

    generation_num: int, default=0
        Current generation number

    Attributes
    ----------
    final_generation_: int
        The generation in which the evolution ended.
    """

    def __init__(
        self,
        population: Union[Population, Subpopulation, List[Subpopulation]],
        statistics: Union[Statistics, List[Statistics]] = None,
        breeder: Breeder = None,
        population_evaluator: PopulationEvaluator = None,
        termination_checker: Union[
            TerminationChecker, List[TerminationChecker]
        ] = None,
        max_generation: int = 100,
        events: Dict[str, Dict[object, Callable]] = None,
        event_names: List[str] = None,
        random_generator: RNG = RNG(),
        random_seed: int = None,
        generation_seed: int = None,
        executor: str = "process",
        max_workers: int = None,
        generation_num: int = 0,
    ):

        ext_event_names = event_names.copy() if event_names is not None else []

        ext_event_names.extend(
            ["init", "evolution_finished", "after_generation"]
        )
        super().__init__(events=events, event_names=ext_event_names)

        self._validate_population_type(population)
        self._validate_statistics_type(statistics)

        self.breeder = breeder
        self.population_evaluator = population_evaluator
        self.termination_checker = termination_checker
        self.max_generation = max_generation

        # set random seed to current time if not provided
        if random_seed is None:
            t = time()
            # convert seed to int for np.random compatibility
            pre_dec_pnt, post_dec_pnt = str(t).split(".")
            int_seed = int(pre_dec_pnt + post_dec_pnt)
            random_seed = int_seed % (2**32)

        self.random_generator = random_generator
        self.random_seed = random_seed
        self.generation_seed = (
            generation_seed if generation_seed is not None else random_seed
        )

        self.best_of_run_ = None
        self.worst_of_gen = None
        self.generation_num = generation_num

        self.max_workers = max_workers

        if executor == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif executor == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            raise ValueError('Executor must be either "thread" or "process"')
        self._executor_type = executor

        self.final_generation_ = 0

    @overrides
    def apply_operator(self, payload):
        """
        begin the evolutionary run
        """
        self.evolve()

    def evolve(self) -> None:
        """
        Performs the evolutionary run by initializing the random seed,
        creating the population, performing the evolutionary loop
        and finally finishing the evolution process
        """
        self.initialize()

        if self.should_terminate(
            self.population, self.best_of_run_, self.generation_num
        ):
            self.final_generation_ = 0
            self.publish("after_generation")
        else:
            self.evolve_main_loop()

        self.publish("evolution_finished")
        self.finish()

    @abstractmethod
    def execute(self, **kwargs) -> object:
        """
        Execute the algorithm result after evolution ended.

        Parameters
        ----------
        kwargs : keyword arguments (relevant in GP representation)
                Input to program, including every variable
                in the terminal set as a keyword argument.
                For example, if `terminal_set=['x', 'y', 'z', 0, 1, -1]`
                then call `execute(x=..., y=..., z=...)`.

        Returns
        -------
        object
                Result of algorithm execution (for example: the best
                 individual in GA, or the best individual execution in GP)
        """
        raise ValueError("execute is an abstract method in class Algorithm")

    def initialize(self) -> None:
        """
        Initialize seed, Executor and relevant operators
        """
        self.set_random_seed(self.random_seed)
        logger.info("random seed = %d", self.random_seed)
        self.population_evaluator.set_executor(self.executor)

        for field in self.__dict__.values():
            if isinstance(field, Operator):
                field.initialize()

        self.create_population()
        self.best_of_run_ = self.population_evaluator.act(self.population)
        self.publish("init")

    def _validate_population_type(self, population: Any) -> None:
        # Assert valid population input
        if population is None:
            raise ValueError("Population cannot be None")

        if isinstance(population, Population):
            self.population = population
        elif isinstance(population, Subpopulation):
            self.population = Population([population])
        elif isinstance(population, list):
            if len(population) == 0:
                raise ValueError("Population cannot be empty")
            for sub_pop in population:
                if not isinstance(sub_pop, Subpopulation):
                    raise ValueError(
                        "Detected a non-Subpopulation "
                        "instance as an element in Population"
                    )
            self.population = Population(population)
        else:
            raise ValueError(
                "Parameter population must be either a Population, "
                "a Subpopulation or a list of Subpopulations. "
                "Received population with unexpected type of",
                type(population),
            )

    def _validate_statistics_type(self, statistics: Any) -> None:
        # Assert valid statistics input
        if isinstance(statistics, Statistics):
            self.statistics = [statistics]
        elif isinstance(statistics, list):
            for stat in statistics:
                if not isinstance(stat, Statistics):
                    raise ValueError(
                        "Expected a Statistics instance as an element"
                        " in Statistics list, but received",
                        type(stat),
                    )
            self.statistics = statistics
        else:
            raise ValueError(
                "Parameter statistics must be either a subclass of Statistics"
                " or a list of subclasses of Statistics.\n"
                "received statistics with unexpected type of",
                type(statistics),
            )

    def evolve_main_loop(self) -> None:
        """
        Performs the evolutionary main loop
        """
        # there was already "preprocessing" generation created - gen #0
        # now create another self.max_generation generations, starting gen #1
        for gen in range(1, self.max_generation + 1):
            self.generation_num = gen
            self.update_gen(gen)

            self.set_generation_seed(self.next_seed())
            self.generation_iteration(gen)
            if self.should_terminate(self.population, self.best_of_run_, gen):
                self.final_generation_ = gen
                self.publish("after_generation")
                break
            self.publish("after_generation")

    def update_gen(self, gen: int) -> None:
        """
        Update `gen` field for all individuals

        Parameters
        ----------
        gen : int
            Current generation number
        """
        for subpopulation in self.population.sub_populations:
            for ind in subpopulation.individuals:
                ind.gen = gen

    @abstractmethod
    def generation_iteration(self, gen: int) -> bool:
        """
        Performs an iteration of the evolutionary main loop

        Parameters
        ----------
        gen: int
            current generation number

        Returns
        -------
        bool
            True if the main loop should terminate, False otherwise
        """
        raise ValueError(
            "generation_iteration is an abstract method in class Algorithm"
        )

    def finish(self) -> None:
        """
        Finish the evolutionary run
        """
        self.executor.shutdown()

    def create_population(self) -> None:
        """
        Create the population for the evolutionary run
        """
        self.population.create_population_individuals()

    def event_name_to_data(self, event_name) -> Dict[str, object]:
        """
        Convert event name to relevant data of the Algorithm for the event

        Parameters
        ----------
        event_name: string
            name of the event that is happening

        Returns
        ----------
        Dict[str, object]
            Algorithm data regarding the given event
        """
        if event_name == "init":
            return {
                "population": self.population,
                "statistics": self.statistics,
                "breeder": self.breeder,
                "termination_checker": self.termination_checker,
                "max_generation": self.max_generation,
                "events": self.events,
                "max_workers": self.max_workers,
            }
        return {}

    def set_random_seed(self, seed: int = None) -> None:
        """
        Set the initial seed for the random generator
        This method is called once at the beginning of the run.

        Parameters
        ----------
        seed: int
                random seed number
        """
        self.random_generator.set_seed(seed)
        self.random_seed = seed

    def set_generation_seed(self, seed: int) -> None:
        """
        Set the seed for current generation.
        This method is called once every generation.

        Parameters
        ----------
        seed: int
                current generation seed
        """
        self.random_generator.set_seed(seed)
        self.generation_seed = seed

    def next_seed(self) -> int:
        """
        Increase the random seed for the next generation.

        Returns
        ----------
        int
        random seed number
        """
        return (self.generation_seed + 1) % (2**32)

    def should_terminate(
        self,
        population: Population,
        best_of_run_: Individual,
        generation_num: int,
    ) -> bool:
        if self.termination_checker is None:
            return False
        elif isinstance(self.termination_checker, list):
            return any(
                [
                    t.should_terminate(
                        population, best_of_run_, generation_num
                    )
                    for t in self.termination_checker
                ]
            )
        else:
            return self.termination_checker.should_terminate(
                population, best_of_run_, generation_num
            )

    def __getstate__(self) -> Dict[str, object]:
        """
        Return a dictionary of the Algorithm's fields and values.
        It is mainly used for serialization.
        We remove executor field since it cannot be pickled.

        Returns
        -------
        Dict[str, object]
            Dictionary of {field name: field value} for the Algorithm object.
        """
        state = self.__dict__.copy()
        del state["executor"]
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        """
        Set the __dict__ of the algorithm upon deserialization.
        We update executor field according to the _executor_type field,
        since the executor was removed in the serialization process.

        Parameters
        ----------
        state : Dict[str, object]
            Dictionary of {field name: field value} for the Algorithm object.
        """
        self.__dict__.update(state)
        if self._executor_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
