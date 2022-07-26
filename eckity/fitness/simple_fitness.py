from overrides import overrides
"""
This module implements the class `SimpleFitness`
"""

from eckity.fitness.fitness import Fitness


class SimpleFitness(Fitness):
    """
    This class is responsible for handling the fitness score of some Individual
    (checking if fitness is evaluated, comparing fitness scores with other individuals etc.)

    In the simple case, each individual holds a float fitness score

    fitness: float
        the fitness score of an individual

    higher_is_better: bool
        declares the fitness direction.
        i.e., if it should be minimized or maximized
    """
    def __init__(self,
                 fitness: float = None,
                 higher_is_better=False):
        is_evaluated = fitness is not None
        super().__init__(higher_is_better=higher_is_better, is_evaluated=is_evaluated)
        self.fitness: float = fitness

    def set_fitness(self, fitness):
        """
        Updates the fitness score to `fitness`

        Parameters
        ----------
        fitness: float
            the fitness score to be updated
        """
        if self._is_evaluated:
            raise AttributeError('fitness already evaluated and set to', self.fitness)
        self.fitness = fitness
        self._is_evaluated = True

    @overrides
    def get_pure_fitness(self):
        """
        Returns the pure fitness score of the individual (before applying balancing methods like bloat control)

        Returns
        ----------
        float
            fitness score of the individual
        """
        if not self._is_evaluated:
            raise ValueError('Fitness not evaluated yet')
        return self.fitness

    @overrides
    def set_not_evaluated(self):
        """
        Set this fitness score status to be not evaluated
        """
        super().set_not_evaluated()
        self.fitness = None

    def check_comparable_fitness_scores(self, other_fitness):
        """
        Check if `this` fitness score is comparable to `other_fitness`

        Returns
        ----------
        bool
            True if fitness scores are comparable, False otherwise
        """
        if not isinstance(other_fitness, SimpleFitness):
            raise TypeError('Expected SimpleFitness object in better_than, got', type(other_fitness))
        if not self.is_fitness_evaluated() or not other_fitness.is_fitness_evaluated():
            raise ValueError('Fitness scores must be evaluated before comparison')

    @overrides
    def better_than(self, ind, other_fitness, other_ind):
        """
        Compares between the current fitness of the individual `ind` to the fitness score `other_fitness` of `other_ind`
        In the simple case, compares the float fitness scores of the two individuals

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
            True if this fitness score is better than the `other` fitness score, False otherwise
        """
        self.check_comparable_fitness_scores(other_fitness)
        return self.get_augmented_fitness(ind) > other_fitness.get_augmented_fitness(other_ind) \
            if self.higher_is_better \
            else self.get_augmented_fitness(ind) < other_fitness.get_augmented_fitness(other_ind)

    @overrides
    def equal_to(self, ind, other_fitness, other_ind):
        """
        Compares between the current fitness of the individual `ind` to the fitness score `other_fitness` of `other_ind`
        In the simple case, compares the float fitness scores of the two individuals

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
            True if this fitness score is equal to the `other` fitness score, False otherwise
        """
        self.check_comparable_fitness_scores(other_fitness)
        return self.get_augmented_fitness(ind) == other_fitness.get_augmented_fitness(other_ind)

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.should_cache_between_gens:
            state['_is_evaluated'] = False
            state['fitness'] = None
        return state
