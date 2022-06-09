"""
This module implements the vector class.
"""

import numpy as np
from numbers import Number
from random import randint, uniform, random

from eckity.base.utils import arity
from eckity.genetic_encodings.gp.tree.utils import _generate_args

from eckity.individual import Individual
from eckity.genetic_encodings.gp.tree.functions import f_add, f_sub, f_mul, f_div


class FloatVector(Vector):
    def __init__(self,
                 fitness,
                 length=1,
                 bounds=(0.0, 1.0)):
        super().__init__(fitness, length, bounds)

        def get_random_number_in_bounds(self, index):
            if type(bounds) == tuple:
                return random.uniform(bounds[0], vector.get_bounds[1])
            return random.uniform(bounds[index][0], vector.get_bounds[index][1])

        def get_random_number_with_gauss(self, index, gauss):
            return index  # index

# end class float vector