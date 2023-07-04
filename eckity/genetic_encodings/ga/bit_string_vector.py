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
        Fitness handler class.
        Responsible of keeping the fitness value of the individual.

    length : int
        Vector length - the number of cells in the vector.

    bounds : tuple or list of tuples
        Min/Max values for each vector cell (if of length n),
        or the minimum and maximum (if of length 1).
    """

    def __init__(self,
                 fitness,
                 length,
                 bounds=(0, 1),
                 vector=None):
        super().__init__(fitness=fitness,
                         length=length,
                         bounds=bounds,
                         vector=vector)

    def get_random_number_in_bounds(self, index):
        """
        Return a random number of available cell values (0 or 1),
        with equal probability.

        Parameters
        ----------
        index : int
            cell index

        Returns
        -------
        int
            random value according to bounds field
        """
        return randint(self.bounds[0], self.bounds[1])

    def bit_flip(self, index):
        """
        Flip the bit in the given index.
        """
        return self.bounds[1] \
            if self.cell_value(index) == self.bounds[0] else self.bounds[0]

# end class bit string vector
