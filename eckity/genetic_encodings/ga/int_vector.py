"""
This module implements the IntVector class.
"""

from random import randint

from eckity.genetic_encodings.ga.vector_individual import Vector

MIN_BOUND = 2 ** 31 - 1
MAX_BOUND = -2 ** 31


class IntVector(Vector):
    """
    An Integer Vector individual representation for Genetic Algorithms operations.
    It is represented by a list of integers.

    Parameters
    ----------
    fitness : Fitness
        Fitness handler class, responsible of keeping the fitness value of the individual.

    length : int
        Vector length - the number of cells in the vector.

    bounds : list of tuples
        Min/Max values for each vector cell (if of length n), or the minimum and maximum (if of length 1).
    """
    def __init__(self,
                 fitness,
                 length,
                 bounds=(MIN_BOUND, MAX_BOUND)):
        super().__init__(fitness, length, bounds)

    def get_random_number_in_bounds(self, index):
        """
        Return a random number from possible cell values, according to bounds.

        Parameters
        ----------
        index : int
            cell index

        Returns
        -------
        float
            random value according to bounds field
        """
        if type(self.bounds) == tuple:
            return randint(self.bounds[0], self.bounds[1])
        return randint(self.bounds[index][0], self.bounds[index][1])

# end class int vector
