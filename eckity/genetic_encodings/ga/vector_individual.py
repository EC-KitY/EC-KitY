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


class AbstractVector(Individual):
    """
    A tree optimized for genetic programming operations.
    It is represented by a list of nodes in depth-first order.
    There are two types of nodes: functions and terminals.

    (tree is not meant as a stand-alone -- parameters are supplied through the call from the Tree Creators)

    Parameters
    ----------
    init_depth : (int, int)
        Min and max depths of initial random trees. The default is None.

    function_set : list
        List of functions used as internal nodes in the gp tree. The default is None.

    terminal_set : list
        List of terminals used in the gp-tree leaves. The default is None.

    erc_range : (float, float)
        Range of values for ephemeral random constant (erc). The default is None.
    """

    def __init__(self,
                 fitness,
                 bounds,
                 length=1):
        super().__init__(fitness)

        assert (type(bounds) == tuple and len(bounds == 2)) or (type(bounds) == array and len(bounds) == length)
        self.bounds = bounds  # todo remove cell_range
        self.length = length
        self.vector = []

    def size(self):
        """
        Compute size of vector.

        Returns
        -------
        int
            vector size (= number of cells).
        """
        return len(self.vector)

    def get_bounds(self):
        return self.bounds

    def add_cell(self, cell):
        self.vector.append(cell)

    def empty_vector(self):
        self.vector = []

    def set_vector(self, vector):
        self.vector = vector

    def get_vector(self):
        return self.vector

    def random_vector_part(self):
        # select a random node index from this individual's vector
        rnd_i = randint(0, self.size() - 1)
        # select the end
        end_i = randint(rnd_i, self.size() - 1)  # todo check
        # now we have a random subtree from this individual
        return self.vector[rnd_i:end_i + 1]

    def replace_vector_part_random(self, vector):
        """
        Replace the subtree starting at `index` with `subtree`

        Parameters
        ----------
        subtree - new subtree to replace the some existing subtree in this individual's tree

        Returns
        -------
        None
        """
        index = randint(0, self.size() - len(vector))  # select a random node (index)
        end_i = index + len(vector)
        replaced_part = self.vector[index:end_i]
        self.vector = self.vector[index:].extend(vector).extend(self.vector[:end_i])  # todo check
        return replaced_part

    def replace_vector_part(self, vector, start_index):
        """
        Replace the subtree starting at `index` with `subtree`

        Parameters
        ----------
        subtree - new subtree to replace the some existing subtree in this individual's tree

        Returns
        -------
        None
        """
        end_i = start_index + len(vector)
        replaced_part = self.vector[index:end_i]
        self.vector = self.vector[start_index:].extend(vector).extend(self.vector[:end_i])  # todo check
        return replaced_part

    def get_vector_part(self, index, end_i):
        return self.vector[index:end_i]

    def _cell_value(self, index):
        return self.vector[index]

    def set_cell_value(self, index, value):
        self.vector[index] = value

    def get_random_number_in_bounds(self):
        raise Exception("get_random_number is abs method in vector individual")

    def show(self):
        """
        Print out a simple textual representation of the vector.

        Returns
        -------
        None.
        """
        print(self.vector)

    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, Vector) \
               and self.function_set == other.function_set \
               and self.terminal_set == other.terminal_set \
               and self.n_terminals == other.n_terminals \
               and self.arity == other.arity \
               and self.vars == other.vars \
               and self.erc_range == other.erc_range \
               and self.n_functions == other.n_functions \
               and self.init_depth == other.init_depth \
               and self.tree == other.tree  # todo do we need this? refactor

# end class tree
