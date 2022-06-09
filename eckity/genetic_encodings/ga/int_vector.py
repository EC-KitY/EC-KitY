"""
This module implements the vector class.
"""

import sys
from random import randint

from eckity.genetic_encodings.ga.vector_individual import Vector

MIN_BOUND = 2 ** 31 - 1
MAX_BOUND = -2 ** 31


class IntVector(Vector):
    def __init__(self,
                 fitness,
                 length,
                 bounds=(MIN_BOUND, MAX_BOUND)):
        super().__init__(fitness, length, bounds)

    def get_random_number_in_bounds(self, index):
        if type(self.bounds) == tuple:
            return randint(self.bounds[0], self.bounds[1])
        return randint(self.bounds[index][0], self.bounds[index][1])

# end class int vector
