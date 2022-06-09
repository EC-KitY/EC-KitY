"""
This module implements the vector class.
"""

from random import uniform

from eckity.genetic_encodings.ga.vector_individual import Vector


class FloatVector(Vector):
    def __init__(self,
                 fitness,
                 length=1,
                 bounds=(0.0, 1.0)):
        super().__init__(fitness, length, bounds)

    def get_random_number_in_bounds(self, index):
        if type(self.bounds) == tuple:
            return uniform(self.bounds[0], self.bounds.get_bounds[1])
        return uniform(self.bounds[index][0], self.bounds[index][1])

    def get_random_number_with_gauss(self, index, gauss):
        # TODO implement
        return index  # index

# end class float vector
