"""
This module implements the class `Fitness`
"""

from abc import abstractmethod, ABC


class Fitness(ABC):
    """
    This class is responsible for handling the fitness score of some Individual
    (checking if fitness is evaluated, comparing fitness scores with other individuals etc.)

    is_evaluated: bool
        declares if fitness score is evaluated and updated in the current generation

    higher_is_better: bool
        declares the fitness direction.
        i.e., if it should be minimized or maximized

    cache: bool
        declares whether the fitness score should reset at the end of each generation

    is_relative_fitness: bool
        declares whether the fitness score is absolute or relative
    """

    def __init__(
        self,
        is_evaluated: bool = False,
        higher_is_better: bool = None,
        is_relative_fitness: bool = False,
        cache: bool = False,
    ):
        self._is_evaluated = is_evaluated
        self.is_relative_fitness = is_relative_fitness
        self.cache = False if is_relative_fitness else cache

        if higher_is_better is None:
            raise ValueError("higher_is_better must be set to True/False")
        self.higher_is_better = higher_is_better

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
