from random import sample

from eckity.genetic_operators.genetic_operator import GeneticOperator


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
        self.points = sorted(sample(range(0, individuals[0].size()), self.k))

        start_index = 0
        for end_point in self.points:
            replaced_part = individuals[0].get_vector_part(start_index, end_point)
            replaced_part = individuals[1].replace_vector_part(replaced_part, start_index)
            individuals[0].replace_vector_part(replaced_part, start_index)
            start_index = end_point  # todo add last iter

        self.applied_individuals = individuals
        return individuals
