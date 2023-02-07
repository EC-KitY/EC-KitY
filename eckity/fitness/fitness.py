"""
This module implements the class `Fitness`
"""

from abc import abstractmethod


class Fitness:
    """
    This class is responsible for handling the fitness score of some Individual
    (checking if fitness is evaluated, comparing fitness scores with other individuals etc.)

    context: list of Individuals
        individuals involved in calculating the fitness (co-evolution)

    trials: list of floats
        fitness results for previous trials done to calculate fitness (co-evolution)

    _is_evaluated: bool
        declares if fitness score is evaluated and updated in the current generation

    is_relative_fitness: bool
        declares whether the fitness score is absolute or relative

    should_cache_between_gens: bool
        declares whether the fitness score should reset at the end of each generation

    higher_is_better: bool
        declares the fitness direction.
        i.e., if it should be minimized or maximized
    """
    def __init__(self,
                 context=None,
                 trials=None,
                 is_evaluated=False,
                 is_relative_fitness=False,
                 should_cache_between_gens=False,
                 higher_is_better=False):
        self.context = context
        self.trials = trials
        self._is_evaluated = is_evaluated
        self.is_relative_fitness = is_relative_fitness
        self.should_cache_between_gens = False if is_relative_fitness else should_cache_between_gens
        self.higher_is_better = higher_is_better
        self.optimal_fitness = 1 if higher_is_better else 0

    @abstractmethod
    def get_pure_fitness(self):
        """
        Returns the pure fitness score of the individual (before applying balancing methods like bloat control)
        """
        raise ValueError("get_pure_fitness is an abstract method in class Fitness")

    def get_augmented_fitness(self, individual):
        """
        Returns the fixed fitness score of the individual (after applying balancing methods like bloat control)
        By default, returns the pure fitness score

        Parameters
        ----------
        individual: Individual
            the individual instance that holds this Fitness instance

        Returns
        ----------
        object
            Fixed fitness value for the given individual
        """
        return self.get_pure_fitness()

    @abstractmethod
    def better_than(self, ind, other_fitness, other_ind):
        """
        Compares between the current fitness of the individual `ind` to the fitness score `other_fitness` of `other_ind`

        Parameters
        ----------
        ind: Individual
            the individual instance that holds this Fitness instance

        other_fitness: Fitness
            the Fitness instance of the `other` individual

        other_ind: Individual
            the `other` individual instance which is being compared to the individual `ind`

        Returns
        ----------
        bool
            True if this fitness is better than the `other` fitness, False otherwise
        """
        raise ValueError("better_than is an abstract method in class Fitness")

    @abstractmethod
    def equal_to(self, ind, other_fitness, other_ind):
        """
        Compares between the current fitness of the individual `ind` to the fitness score `other_fitness` of `other_ind`

        Parameters
        ----------
        ind: Individual
            the individual instance that holds this Fitness instance

        other_fitness: Fitness
            the Fitness instance of the `other` individual

        other_ind: Individual
            the `other` individual instance which is being compared to the individual `ind`

        Returns
        ----------
        bool
            True if this fitness is equal to the `other` fitness, False otherwise
        """
        raise ValueError("better_than is an abstract method in class Fitness")

    def set_not_evaluated(self):
        """
        Set this fitness score status to be not evaluated
        """
        if not self.is_fitness_evaluated():
            raise ValueError('Fitness already not evaluated')
        self._is_evaluated = False

    def is_fitness_evaluated(self):
        """
        Check this fitness score status (if the fitness score is updated)

        Returns
        ----------
        bool
            True if this fitness is evaluated, False otherwise
        """
        if self.is_relative_fitness:
            return True
        return self._is_evaluated
