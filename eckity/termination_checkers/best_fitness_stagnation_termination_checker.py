import math

from eckity.termination_checkers.termination_checker import TerminationChecker


class BestFitnessStagnationTerminationChecker(TerminationChecker):
    """
    Concrete Termination Checker that checks that best firnes.

    Parameters
    ----------
    stagnation_generations: int, default=100.
        This termination checker checks if the best fitness hasn't changed for stagnation_generations generations.
    """

    def __init__(self, stagnation_generations_to_terminate=100):
        super().__init__()
        self.stagnation_generations_to_terminate = stagnation_generations_to_terminate
        self.best_fitnesses = None
        self.stagnation_generations = 0

    def should_terminate(self, population, best_individual, gen_number):
        """
        Determines if the best fitness hasn't changed for stagnation_generations generations.
        If so, recommends the algorithm to terminate early.

        Parameters
        ----------
        population: Population
            The evolutionary experiment population of individuals.

        best_individual: Individual
            The individual that has the best fitness of the algorithm.

        gen_number: int
            Current generation number.

        Returns
        -------
        bool
            True if the algorithm should terminate early, False otherwise.
        """
        best_fitnesses = [ind.get_pure_fitness() for ind in population.get_best_individuals()]
        if self.best_fitnesses and all([math.isclose(a, b) for a, b in zip(best_fitnesses, self.best_fitnesses)]):
            self.stagnation_generations += 1
            return self.stagnation_generations >= self.stagnation_generations_to_terminate
        else:
            self.stagnation_generations = 0
            self.best_fitnesses = best_fitnesses
            return False
