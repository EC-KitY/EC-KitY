"""
This module implements the FloatVector class.
"""

from random import uniform, gauss

from eckity.genetic_encodings.ga.vector_individual import Vector


class FloatVector(Vector):
    """
    A Float Vector individual representation for Genetic Algorithms operations.
    It is represented by a list of floats.

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
                 length=1,
                 bounds=(0.0, 1.0)):
        super().__init__(fitness, length, bounds)

    def get_random_number_in_bounds(self, index):
        """
        Return a random number from possible cell values.

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
            return uniform(self.bounds[0], self.bounds.get_bounds[1])
        return uniform(self.bounds[index][0], self.bounds[index][1])

    def get_random_number_with_gauss(self, index, mu, sigma):
        """
        Return a random number from possible cell values, with an addition of gaussian noise.

        Parameters
        ----------
        index : int
            cell index
        mu : float
            gaussian mean
        sigma : float
            gaussian standard deviation

        Returns
        -------
        float
            random value according to bounds field and gauss parameters
        """
        return self.cell_value(index) + gauss(mu, sigma)

# end class float vector
