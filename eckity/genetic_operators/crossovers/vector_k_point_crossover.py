from random import sample

from eckity.genetic_operators.genetic_operator import GeneticOperator
from eckity.genetic_encodings.ga.vector_individual import Vector


class VectorKPointsCrossover(GeneticOperator):
    def __init__(self, probability=1, arity=2, k=1, events=None):
        """
            Vector N Point Mutation.

            Randomly chooses N vector cells and performs a small change in their values.

            Parameters
            ----------
            probability : float
                The probability of the mutation operator to be applied

            arity : int
                The number of individuals this mutation is applied on

            k : int
                Number of points to cut the vector for the crossover.

            events: list of strings
                Events to publish before/after the mutation operator
        """
        self.individuals = None
        self.applied_individuals = None
        self.k = k
        self.points = None
        super().__init__(probability=probability, arity=arity, events=events)

    def apply(self, individuals):
        """
        Attempt to perform the mutation operator

        Parameters
        ----------
        individuals : list of individuals
            individuals to perform crossover on

        Returns
        ----------
        list of individuals
            individuals after the crossover
        """
        self.individuals = individuals
        self.points = sorted(sample(range(1, individuals[0].size()), self.k))
        
        start_index = 0
        for i in range(0, len(self.points), 2):
            end_point = self.points[i]
            replaced_part = individuals[0].get_vector_part(start_index, end_point)
            replaced_part = individuals[1].replace_vector_part(replaced_part, start_index)
            individuals[0].replace_vector_part(replaced_part, start_index)
            start_index = end_point

        # replace the last part (from last point to end)
        replaced_part = individuals[0].get_vector_part(self.points[-1], individuals[0].size())
        replaced_part = individuals[1].replace_vector_part(replaced_part, self.points[-1])
        individuals[0].replace_vector_part(replaced_part, self.points[-1])

        self.applied_individuals = individuals
        return individuals
