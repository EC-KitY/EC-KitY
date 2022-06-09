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


class BitStringVector(Vector):
    def __init__(self,
                 fitness,
                 length,
                 bounds=(0, 1)):
        super().__init__(fitness, length, bounds)

    def get_random_number_in_bounds(self):
        return randint(individual.get_bounds[0], vector.get_bounds[1]) #todo check if need to check bounds is it tuple always




# end class bit string vector
