"""
This module implements the BitStringVector class.
"""

from random import randint

from eckity.genetic_encodings.ga.vector_individual import Vector


class BitStringVector(Vector):
    """
    A Bit Vector individual representation for Genetic Algorithms operations.
    It is represented by a list of ones and zeros.

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
                 bounds=(0, 1)):
        super().__init__(fitness=fitness, length=length, bounds=bounds)

    def get_random_number_in_bounds(self, index):
        """
        Return a random number of available cell values - 0 or 1, with equal probability.

        Parameters
        ----------
        index : int
            cell index

        Returns
        -------
        int
            random value according to bounds field
        """
        # todo check if need to check bounds - is it tuple always?
        return randint(self.bounds[0], self.bounds[1])

    def bit_flip(self, index):
        """
        Return a random number of available cell values - 0 or 1, with equal probability.

        Parameters
        ----------
        index : int
            cell index

        Returns
        -------
        int
            random value according to bounds field
        """
        return self.bounds[1] if self.cell_value(index) == self.bounds[0] else self.bounds[0]

# end class bit string vector
