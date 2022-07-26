"""
This module implements the `GPFitness` class
"""

from overrides import overrides

from eckity.fitness.simple_fitness import SimpleFitness


class GPFitness(SimpleFitness):
    """
    This class is responsible for handling the fitness score of some Individual
    (checking if fitness is evaluated, comparing fitness scores with other individuals etc.)

    In the simple case, each individual holds a float fitness score
    GPFitness also adds bloat control to the fitness score, by "punishing" the fitness score of large trees

    fitness: float
        the fitness score of an individual

    higher_is_better: bool
        declares the fitness direction.
        i.e., if it should be minimized or maximized

    bloat_weight: float
        the weight of the bloat control fitness reduction
    """
    def __init__(self,
                 fitness: float = None,
                 higher_is_better=False,
                 bloat_weight=0.1):
        super().__init__(fitness=fitness, higher_is_better=higher_is_better)
        self.bloat_weight = bloat_weight

    @overrides
    def get_augmented_fitness(self, individual):
        """
        Returns the fixed fitness of a given individual, after including bloat control

        Parameters
        ----------
        individual: Individual
            a GP Tree to apply bloat control on

        Returns
        ----------
        float
            augmented fitness score after applying bloat control
        """
        if not self.is_fitness_evaluated():
            raise ValueError('Fitness not evaluated yet')

        # subtract bloat value from the fitness score if it should be maximized,
        # otherwise add bloat value to fitness score
        return self.fitness - self.bloat_weight * individual.size() \
            if self.higher_is_better \
            else self.fitness + self.bloat_weight * individual.size()
